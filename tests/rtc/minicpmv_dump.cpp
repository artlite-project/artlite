// Golden tensors of llama.cpp's MiniCPM-V 4.7 vision path (libmtmd): an image prompt through the projector and the
// language model on the CPU backend, with the vision encoder's intermediate tensors dumped as .npy files.
//
// Built against a prebuilt llama.cpp tree (include/, ggml/include/, tools/mtmd/, build/bin/libmtmd.so) by
// minicpmv_ref.py.
//
//   minicpmv_dump -m MODEL --mmproj MMPROJ --image FILE -p PROMPT -o OUTDIR [-a ANSWER] [-e EMBD] [-n REGEX]
//                 [--threads N]
//   minicpmv_dump -m MODEL --seq SEQ --pos POS -o OUTDIR [--from K] [--threads N]
//
// PROMPT: a text file with the media marker (mtmd_default_marker()) where the image goes; ANSWER: raw little-endian
// int32 token ids appended after the prompt (teacher forcing); EMBD: raw float32 embeddings (n_tokens, n_embd) that the
// language model takes for the image instead of the encoder's (the encoder is not run). REGEX: the encoder tensors
// (names without the "-<il>" suffix) to dump.
//
// --seq: the language model alone on a whole sequence of embedding rows with given M-RoPE positions (the official
// ones, which mtmd's are not): SEQ raw float32 (n, n_embd), the text tokens' rows too; POS raw int32 (n, 3), per token
// (T, H, W). A token's KV position is its index, so the attention is causal in token order; its RoPE sections 0, 1, 2
// turn by T, H, W (the slots z, y, x of an embedding under rope.section_order zyxt). Output: logits.npy (the rows of
// tokens K .. n - 1), chunks.json ({"chunks": [], "logit_tokens", "entries": 0}), meta.json.
//
// Output in OUTDIR:
// - chunks.json: the prompt's chunks in order, text ({"text": [token ids]}) or image ({"image": n_tokens, "n_pos",
//   "pos": [[t, x, y, z] per token]: mtmd_image_tokens_get_decoder_pos at the chunk's first position}),
//   then the answer's tokens as a text chunk; "logit_tokens": the index in the whole sequence of each row of logits;
//   "entries": the encoder's graphs (the image's entries);
// - embd-<i>.npy: image chunk i's embeddings (n_tokens, n_embd), as mtmd_encode_chunk gives them;
// - <name>-<il>.e<k>.npy: the encoder's tensors per entry k of an image (the overview, then the slices row by row);
//   the inputs (inp_raw: the entry's pixels; positions, the merges' gather indices) as the first nodes that read them
//   see them; float tensors as float32, integer ones as int32, shape = ne reversed;
// - logits.npy: (text tokens, vocab) float32: the logits after every text token (the image's tokens take none);
// - meta.json: what ran.
//
// Backend: CPU only, F32 KV cache, flash attention off.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

// A .npy file written at once.
static void write_npy(const std::string & path, const void * data, size_t bytes, const std::vector<int64_t> & shape,
                      const char * dtype) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
        if (i + 1 < shape.size()) {
            sh += " ";
        }
    }
    sh += ")";
    std::string dict = std::string("{'descr': '") + dtype + "', 'fortran_order': False, 'shape': " + sh + ", }";
    std::string head = std::string("\x93NUMPY\x01\x00", 8);
    size_t total = head.size() + 2 + dict.size() + 1;
    size_t pad = (64 - total % 64) % 64;
    dict += std::string(pad, ' ') + "\n";
    const uint16_t n = (uint16_t) dict.size();
    head += (char) (n & 0xFF);
    head += (char) (n >> 8);
    head += dict;
    fwrite(head.data(), 1, head.size(), f);
    fwrite(data, 1, bytes, f);
    fclose(f);
}

struct State {
    std::string outdir;
    std::regex names;
    int entry = -1;                        // the encoder's entry (a graph per entry, from the node reading inp_raw)
    std::map<std::string, int> seen;       // names dumped in the current entry
    std::set<std::string> inputs_seen;     // inputs dumped in the current entry
};

static bool host(const ggml_tensor * t) { return t->buffer && ggml_backend_buffer_is_host(t->buffer); }

// a tensor's elements in numpy order (ne[0] fastest): float types as float32, integer types as int32
static void dump(const State & st, const ggml_tensor * t, const std::string & name) {
    if (!host(t)) {
        fprintf(stderr, "%s: not in host memory (CPU backend only)\n", t->name);
        exit(1);
    }
    const bool is_int = t->type == GGML_TYPE_I32;
    std::vector<float> f;
    std::vector<int32_t> iv;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                const char * row = (const char *) t->data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1];
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const char * p = row + i0 * t->nb[0];
                    switch (t->type) {
                        case GGML_TYPE_F32: f.push_back(*(const float *) p); break;
                        case GGML_TYPE_F16: f.push_back(ggml_fp16_to_fp32(*(const ggml_fp16_t *) p)); break;
                        case GGML_TYPE_BF16: f.push_back(ggml_bf16_to_fp32(*(const ggml_bf16_t *) p)); break;
                        case GGML_TYPE_I32: iv.push_back(*(const int32_t *) p); break;
                        default: return;
                    }
                }
            }
        }
    }
    std::vector<int64_t> shape;
    int top = 3;
    while (top > 0 && t->ne[top] == 1) {
        top--;
    }
    for (int k = top; k >= 0; k--) {
        shape.push_back(t->ne[k]);
    }
    const std::string path = st.outdir + "/" + name + ".e" + std::to_string(st.entry) + ".npy";
    if (is_int) {
        write_npy(path, iv.data(), iv.size() * 4, shape, "<i4");
    } else {
        write_npy(path, f.data(), f.size() * 4, shape, "<f4");
    }
}

static bool float_or_int(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 ||
           t->type == GGML_TYPE_I32;
}

static bool wanted(const State & st, const ggml_tensor * t) {
    if (strpbrk(t->name, " (") || !float_or_int(t)) {
        return false;
    }
    std::string s(t->name);
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        s.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        s = s.substr(0, dash);
    }
    return std::regex_match(s, st.names);
}

// the graph inputs a node reads (inp_raw, positions, gather indices: leafs the callback never sees itself)
static bool input_name(const char * name) {
    return !strcmp(name, "inp_raw") || !strcmp(name, "positions") || !strncmp(name, "vit_merger_", 11) ||
           !strncmp(name, "merger_ds_idx_", 14);
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    State & st = *(State *) user;
    if (ask) {
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
            if (!strcmp(t->src[i]->name, "inp_raw")) {  // (the patch embedding: an entry's graph begins)
                st.entry++;
                st.seen.clear();
                st.inputs_seen.clear();
            }
        }
        bool reads_input = false;
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
            reads_input = reads_input || (input_name(t->src[i]->name) && !st.inputs_seen.count(t->src[i]->name));
        }
        return reads_input || wanted(st, t);
    }
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        const ggml_tensor * s = t->src[i];
        if (input_name(s->name) && !st.inputs_seen.count(s->name) && host(s) && float_or_int(s)) {
            st.inputs_seen.insert(s->name);
            dump(st, s, s->name);
        }
    }
    if (wanted(st, t)) {
        std::string name = t->name;
        const int dup = st.seen[name]++;
        if (dup) {
            name += "." + std::to_string(dup);
        }
        dump(st, t, name);
    }
    return true;
}

static void usage() {
    fprintf(stderr, "usage: minicpmv_dump -m MODEL --mmproj MMPROJ --image FILE -p PROMPT -o OUTDIR [-a ANSWER.i32]\n"
                    "                     [-e EMBD.f32] [-n REGEX] [--threads N]\n"
                    "       minicpmv_dump -m MODEL --seq SEQ.f32 --pos POS.i32 -o OUTDIR [--from K] [--threads N]\n");
    exit(1);
}

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        exit(1);
    }
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    fclose(f);
    return s;
}

static llama_context_params context_params(int n_tokens, int threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((n_tokens + 255) / 256 * 256);
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.no_perf = true;
    return cp;
}

// --seq (the header)
static int run_sequence(llama_model * model, const std::string & seq_path, const std::string & pos_path, int from,
                        const std::string & outdir, int threads) {
    const int n_embd = llama_model_n_embd_inp(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::string seq = read_file(seq_path), pos = read_file(pos_path);
    const int n = (int) (seq.size() / (4 * (size_t) n_embd));
    if (seq.size() != (size_t) n * n_embd * 4 || pos.size() != (size_t) n * 3 * 4 || from < 0 || from >= n) {
        fprintf(stderr, "%s / %s: %zu / %zu bytes, from %d: not (n, %d) float32 / (n, 3) int32\n", seq_path.c_str(),
                pos_path.c_str(), seq.size(), pos.size(), from, n_embd);
        return 1;
    }
    const float * rows = (const float *) seq.data();
    const int32_t * thw = (const int32_t *) pos.data();
    llama_context * ctx = llama_init_from_model(model, context_params(n, threads));
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }
    llama_batch_ext * b = llama_batch_ext_init(ctx);
    for (int i = 0; i < n; i++) {
        const llama_embd e = {rows + (size_t) i * n_embd, 1, (size_t) n_embd};
        const int32_t idx = llama_batch_ext_add_embd(b, 0, e);
        const llama_pos p[4] = {i, thw[3 * i + 1], thw[3 * i + 2], thw[3 * i]};  // (slots t, y, x, z)
        if (idx != i || !llama_batch_ext_set_pos(b, idx, p) || (i >= from && !llama_batch_ext_set_output_logits(b, idx, true))) {
            fprintf(stderr, "cannot add token %d to the batch\n", i);
            return 1;
        }
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
        fprintf(stderr, "llama_process failed\n");
        return 1;
    }
    std::vector<float> logits;
    for (int i = from; i < n; i++) {
        const float * l = llama_get_logits_ith(ctx, i);
        logits.insert(logits.end(), l, l + n_vocab);
    }
    write_npy(outdir + "/logits.npy", logits.data(), logits.size() * 4, {(int64_t) (n - from), n_vocab}, "<f4");
    FILE * js = fopen((outdir + "/chunks.json").c_str(), "w");
    fprintf(js, "{\"chunks\": [], \"logit_tokens\": [");
    for (int i = from; i < n; i++) {
        fprintf(js, "%s%d", i > from ? ", " : "", i);
    }
    fprintf(js, "], \"entries\": 0}\n");
    fclose(js);
    FILE * f = fopen((outdir + "/meta.json").c_str(), "w");
    fprintf(f, "{\"seq\": \"%s\", \"tokens\": %d, \"from\": %d}\n", seq_path.c_str(), n, from);
    fclose(f);
    llama_batch_ext_free(b);
    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    std::string model_path, mmproj_path, image_path, prompt_path, answer_path, embd_path, regex_arg, seq_path, pos_path;
    int from = 0;
    State st;
    int threads = 16;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = next();
        } else if (a == "--mmproj") {
            mmproj_path = next();
        } else if (a == "--image") {
            image_path = next();
        } else if (a == "-p") {
            prompt_path = next();
        } else if (a == "-a") {
            answer_path = next();
        } else if (a == "-e") {
            embd_path = next();
        } else if (a == "-o") {
            st.outdir = next();
        } else if (a == "-n") {
            regex_arg = next();
        } else if (a == "--threads") {
            threads = std::atoi(next().c_str());
        } else if (a == "--seq") {
            seq_path = next();
        } else if (a == "--pos") {
            pos_path = next();
        } else if (a == "--from") {
            from = std::atoi(next().c_str());
        } else {
            usage();
        }
    }
    const bool sequence = !seq_path.empty();
    if (model_path.empty() || st.outdir.empty() || (sequence ? pos_path.empty()
                                                              : mmproj_path.empty() || image_path.empty() || prompt_path.empty())) {
        usage();
    }
    st.names = std::regex(regex_arg.empty() ? "$^" : regex_arg);
    std::vector<llama_token> answer;
    if (!answer_path.empty()) {
        const std::string raw = read_file(answer_path);
        answer.resize(raw.size() / 4);
        memcpy(answer.data(), raw.data(), answer.size() * 4);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = {nullptr};
    mp.devices = no_devices;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 1;
    }
    if (sequence) {
        const int rc = run_sequence(model, seq_path, pos_path, from, st.outdir, threads);
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = false;
    vp.n_threads = threads;
    vp.warmup = false;
    vp.print_timings = false;
    vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vp.cb_eval = eval_cb;
    vp.cb_eval_user_data = &st;
    mtmd_context * vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vp);
    if (!vctx) {
        fprintf(stderr, "cannot load %s\n", mmproj_path.c_str());
        return 1;
    }
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(vctx, image_path.c_str(), false,
                                                                       mtmd_helper_init_opt_default());
    if (!bw.bitmap) {
        fprintf(stderr, "cannot read %s\n", image_path.c_str());
        return 1;
    }
    const std::string prompt = read_file(prompt_path);
    mtmd_input_text text = {prompt.c_str(), prompt.size(), true, true};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = {bw.bitmap};
    if (mtmd_tokenize(vctx, chunks, &text, bitmaps, 1) != 0) {
        fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    int n_total = 0;
    for (size_t c = 0; c < n_chunks; c++) {
        n_total += (int) mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, c));
    }
    n_total += (int) answer.size();

    const llama_context_params cp = context_params(n_total, threads);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    FILE * js = fopen((st.outdir + "/chunks.json").c_str(), "w");
    fprintf(js, "{\"chunks\": [");
    std::vector<float> logits;
    std::vector<int> logit_tokens;
    llama_pos n_past = 0;
    int seq_index = 0;  // (the token index in the whole sequence)
    int n_images = 0;
    auto decode_text = [&](const llama_token * toks, size_t n) {
        llama_batch batch = llama_batch_init((int) n, 0, 1);
        batch.n_tokens = (int) n;
        for (size_t i = 0; i < n; i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = n_past + (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            exit(1);
        }
        for (size_t i = 0; i < n; i++) {
            const float * l = llama_get_logits_ith(ctx, (int) i);
            logits.insert(logits.end(), l, l + n_vocab);
            logit_tokens.push_back(seq_index + (int) i);
        }
        llama_batch_free(batch);
        n_past += (llama_pos) n;
        seq_index += (int) n;
    };
    for (size_t c = 0; c < n_chunks; c++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, c);
        fprintf(js, "%s\n  ", c ? "," : "");
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
            fprintf(js, "{\"text\": [");
            for (size_t i = 0; i < n; i++) {
                fprintf(js, "%s%d", i ? ", " : "", toks[i]);
            }
            fprintf(js, "]}");
            decode_text(toks, n);
            continue;
        }
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
        const size_t n_tok = mtmd_image_tokens_get_n_tokens(it);
        fprintf(js, "{\"image\": %zu, \"n_pos\": %d, \"pos\": [", n_tok, (int) mtmd_image_tokens_get_n_pos(it));
        for (size_t i = 0; i < n_tok; i++) {
            const mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, n_past, i);
            fprintf(js, "%s[%u, %u, %u, %u]", i ? ", " : "", p.t, p.x, p.y, p.z);
        }
        fprintf(js, "]}");
        const int n_embd = llama_model_n_embd_inp(model);
        std::vector<float> given;
        float * embd = nullptr;
        if (!embd_path.empty()) {  // (the image's embeddings given)
            const std::string raw = read_file(embd_path);
            if (raw.size() != n_tok * n_embd * 4) {
                fprintf(stderr, "%s: %zu bytes, the image takes %zu x %d float32\n", embd_path.c_str(), raw.size(), n_tok,
                        n_embd);
                return 1;
            }
            given.resize(n_tok * n_embd);
            memcpy(given.data(), raw.data(), raw.size());
            embd = given.data();
        } else {
            if (mtmd_encode_chunk(vctx, chunk) != 0) {
                fprintf(stderr, "mtmd_encode_chunk failed\n");
                return 1;
            }
            embd = mtmd_get_output_embd(vctx);
        }
        write_npy(st.outdir + "/embd-" + std::to_string(n_images) + ".npy", embd, n_tok * n_embd * 4,
                  {(int64_t) n_tok, n_embd}, "<f4");
        llama_pos new_n_past = 0;
        if (mtmd_helper_decode_image_chunk(vctx, ctx, chunk, embd, n_past, 0, cp.n_batch, &new_n_past, nullptr,
                                           nullptr) != 0) {
            fprintf(stderr, "mtmd_helper_decode_image_chunk failed\n");
            return 1;
        }
        n_past = new_n_past;
        seq_index += (int) n_tok;
        n_images++;
    }
    if (!answer.empty()) {
        fprintf(js, ",\n  {\"text\": [");
        for (size_t i = 0; i < answer.size(); i++) {
            fprintf(js, "%s%d", i ? ", " : "", answer[i]);
        }
        fprintf(js, "]}");
        decode_text(answer.data(), answer.size());
    }
    fprintf(js, "],\n \"logit_tokens\": [");
    for (size_t i = 0; i < logit_tokens.size(); i++) {
        fprintf(js, "%s%d", i ? ", " : "", logit_tokens[i]);
    }
    fprintf(js, "],\n \"entries\": %d}\n", st.entry + 1);
    fclose(js);
    write_npy(st.outdir + "/logits.npy", logits.data(), logits.size() * 4,
              {(int64_t) logit_tokens.size(), n_vocab}, "<f4");
    {
        FILE * f = fopen((st.outdir + "/meta.json").c_str(), "w");
        fprintf(f, "{\"model\": \"%s\", \"mmproj\": \"%s\", \"image\": \"%s\", \"tokens\": %d, \"images\": %d}\n",
                model_path.c_str(), mmproj_path.c_str(), image_path.c_str(), seq_index, n_images);
        fclose(f);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(vctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
