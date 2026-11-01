// Golden tensors of a llama.cpp model run (Qwen3.5 / 3.6 "qwen35" GGUF files): teacher forcing of an exact
// token sequence on the CPU backend, with the intermediate tensors of selected layers dumped as .npy files.
//
// Built against a prebuilt llama.cpp tree (include/, ggml/include/, build/bin/libllama.so) by ref.py.
//
//   qwen35_dump -m MODEL -t TOKENS -o OUTDIR [-u UBATCH] [--split P] [-l LAYERS] [-n REGEX]
//               [--logits f32|f16|none] [--states last|all] [--threads N] [--mtp MTP_MODEL]
//               [--dflash DFLASH_MODEL --dflash-at P [--dflash-names REGEX] [--dflash-random]]
//
// TOKENS: raw little-endian int32 token ids. The tokens run in ubatches of UBATCH (default: all of them in
// one); --split P: the first P in ubatches of UBATCH, then the rest one per ubatch (prompt, then decode
// steps), in one run. LAYERS: comma-separated layer indices, "g" for the tensors without a layer (il = -1),
// "all" for every layer. REGEX: the tensor names (without the "-<il>" suffix) to dump.
//
// Output: OUTDIR/<name>-<il>.npy (float32, I32 tensors such as the experts' ids ffn_moe_topk int32; ggml tensors as
// numpy arrays: shape = ne reversed): stacked over the run's tokens along the tensor's token axis (the lowest ne[d],
// d >= 1, of the ubatch's token count with all dimensions above it 1) -> (tokens, ...); the recurrent states
// (STATE_NAMES) have no token axis: the last ubatch's (--states last) or all of them (--states all: (ubatches,
// ...)). A name computed twice in a ubatch (Kcur before and after RoPE) gets ".1", ".2", ... from the second on.
// OUTDIR/logits.npy: (tokens, vocab) of every position. OUTDIR/meta.json: what ran.
//
// --mtp: then the model's multi-token-prediction block (an MTP file, or MODEL again when it holds a block of its own:
// Qwen3.8; llama.cpp's LLAMA_CONTEXT_TYPE_MTP context) over the same tokens, positions and ubatches, as llama.cpp's draft-mtp keeps its cache: the row at position
// p takes token p and the model's output-normed hidden state h_nextn of row p - 1 (zeros for p = 0); its tensors of
// LAYERS / REGEX (the MTP layer's index, "g") as OUTDIR/mtp.<name>-<il>.npy, its logits as OUTDIR/mtp.logits.npy.
//
// --dflash: then the DFlash2 drafter as llama.cpp's draft-dflash runs it (``common/speculative.cpp``'s
// common_speculative_impl_draft_dflash): the model's rows 0 .. P - 1 injected (their target layers' inputs, in ubatches of
// UBATCH), then the block at P: tokens[P] and mask tokens at P .. P + block_size - 1, non-causal. OUTDIR/dflash.taps.npy:
// (P, taps, hidden); its tensors matching REGEX (of every layer and without one) as OUTDIR/dflash.<name>-<il>.npy, stacked
// over the rows of each graph (the injection's, the block's); OUTDIR/dflash.lattice.npy: (block_size, hidden): per block
// position the candidates' ids (as float32 values), then their scores [predecessor][candidate]. --dflash-random: the taps
// are normal numbers (1 / 16 of unit variance, the model does not run: for the attention past the window).
//
// Backend: CPU only (no devices), F32 KV cache, flash attention off.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-ext.h"
#include "llama.h"

#include <sys/resource.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

static const std::set<std::string> STATE_NAMES = {
    "new_state", "state_predelta", "conv_states", "conv_states_reshaped", "conv_state_last", "conv_input",
    "dnet_add_ar_state",
};

// A .npy file written in pieces: its header is rewritten with the final shape at close.
struct Npy {
    static constexpr long HEADER = 128;
    FILE * f = nullptr;
    std::vector<int64_t> tail;  // shape after the leading axis
    int64_t rows = 0;           // the leading axis
    std::string dtype = "<f4";
    int elem = 4;

    void open(const std::string & path) {
        f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            exit(1);
        }
        std::vector<char> pad(HEADER, ' ');
        write(pad.data(), HEADER);
    }

    // a short write (a full disk) fails the run: a truncated file must not be cached as a golden
    void write(const void * data, size_t bytes) {
        if (fwrite(data, 1, bytes, f) != bytes) {
            fprintf(stderr, "short write (%zu bytes)\n", bytes);
            exit(1);
        }
    }

    void close() {
        std::string shape = "(" + std::to_string(rows) + (tail.empty() ? "," : "");
        for (int64_t d : tail) {
            shape += ", " + std::to_string(d);
        }
        shape += ")";
        std::string dict = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + shape + ", }";
        std::string head = std::string("\x93NUMPY\x01\x00", 8);
        const int n = (int) (HEADER - head.size() - 2);
        dict.resize(n - 1, ' ');
        dict += '\n';
        head += (char) (n & 0xFF);
        head += (char) (n >> 8);
        head += dict;
        fseek(f, 0, SEEK_SET);
        write(head.data(), head.size());
        if (fclose(f) != 0) {
            fprintf(stderr, "cannot close a .npy file\n");
            exit(1);
        }
        f = nullptr;
    }
};

struct State {
    std::string outdir;
    std::regex names;
    std::set<int> layers;
    bool all_layers = false;
    bool states_all = false;
    int n_tokens = 0;  // tokens of the current ubatch
    int ubatch = 0;    // index of the current ubatch
    std::map<std::string, int> seen;  // names computed in the current ubatch
    std::map<std::string, Npy> files;
    std::map<std::string, std::pair<std::vector<int64_t>, std::vector<float>>> last;  // states: the last capture
    std::string prefix;              // of the files' names
    bool capture_h = false;          // keep h_nextn of every token (--mtp: the MTP block's inputs)
    std::vector<float> h, h_result;  // (h_nextn; result_norm: the same tensor when every row is an output)
};

// "attn_norm-3" -> ("attn_norm", 3); "result_norm" -> ("result_norm", -1)
static std::pair<std::string, int> split_name(const char * full) {
    std::string s(full);
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        std::all_of(s.begin() + dash + 1, s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return {s.substr(0, dash), std::atoi(s.c_str() + dash + 1)};
    }
    return {s, -1};
}

static bool wanted(const State & st, const ggml_tensor * t) {
    if (strpbrk(t->name, " (") || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16 &&
                                    t->type != GGML_TYPE_I32)) {
        return false;  // (ggml's derived names "x (view)", other integer tensors)
    }
    auto [base, il] = split_name(t->name);
    if (!std::regex_match(base, st.names)) {
        return false;
    }
    return il < 0 ? st.layers.count(-1) > 0 : (st.all_layers || st.layers.count(il) > 0);
}

// the tensor's elements as float32 in numpy order (ne[0] fastest); an I32 tensor's as their int32 bit patterns
static std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        fprintf(stderr, "%s: not in host memory (CPU backend only)\n", t->name);
        exit(1);
    }
    std::vector<float> out((size_t) ggml_nelements(t));
    size_t i = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                const char * row = (const char *) t->data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1];
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const char * p = row + i0 * t->nb[0];
                    switch (t->type) {
                        case GGML_TYPE_F32: out[i++] = *(const float *) p; break;
                        case GGML_TYPE_F16: out[i++] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p); break;
                        case GGML_TYPE_BF16: out[i++] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) p); break;
                        case GGML_TYPE_I32: memcpy(&out[i++], p, 4); break;
                        default:
                            fprintf(stderr, "%s: tensor type %s\n", t->name, ggml_type_name(t->type));
                            exit(1);
                    }
                }
            }
        }
    }
    return out;
}

static bool is_h(const ggml_tensor * t) { return strcmp(t->name, "h_nextn") == 0 || strcmp(t->name, "result_norm") == 0; }

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    State & st = *(State *) user;
    if (ask) {
        return wanted(st, t) || (st.capture_h && is_h(t));
    }
    if (st.capture_h && is_h(t)) {
        const std::vector<float> v = to_f32(t);
        std::vector<float> & h = strcmp(t->name, "h_nextn") == 0 ? st.h : st.h_result;
        h.insert(h.end(), v.begin(), v.end());
    }
    if (!wanted(st, t)) {
        return true;
    }
    std::string name = t->name;
    const int dup = st.seen[name]++;
    if (dup) {
        name += "." + std::to_string(dup);
    }
    const std::vector<float> v = to_f32(t);
    auto [base, il] = split_name(t->name);
    std::vector<int64_t> rev = {t->ne[3], t->ne[2], t->ne[1], t->ne[0]};
    if (STATE_NAMES.count(base)) {
        if (st.states_all) {
            Npy & f = st.files[name];
            if (!f.f) {
                f.open(st.outdir + "/" + st.prefix + name + ".npy");
                f.tail = rev;
            }
            f.write(v.data(), v.size() * 4);
            f.rows++;
        } else {
            st.last[name] = {rev, v};
        }
        return true;
    }
    int d = 0;  // the token axis
    for (int k = 1; k <= 3 && !d; k++) {
        bool above = true;
        for (int j = k + 1; j <= 3; j++) {
            above = above && t->ne[j] == 1;
        }
        if (t->ne[k] == st.n_tokens && above) {
            d = k;
        }
    }
    Npy & f = st.files[name];
    if (!f.f) {
        f.open(st.outdir + "/" + st.prefix + name + ".npy");
        if (t->type == GGML_TYPE_I32) {
            f.dtype = "<i4";
        }
        if (d) {
            for (int k = d - 1; k >= 0; k--) {
                f.tail.push_back(t->ne[k]);
            }
        } else {  // no token axis: one capture per ubatch
            f.tail = rev;
        }
    }
    f.write(v.data(), v.size() * 4);
    f.rows += d ? st.n_tokens : 1;
    return true;
}

static void usage() {
    fprintf(stderr,
            "usage: qwen35_dump -m MODEL -t TOKENS.i32 -o OUTDIR [-u UBATCH] [--split P] [-l LAYERS] [-n REGEX]\n"
            "                   [--logits f32|f16|none] [--states last|all] [--threads N] [--mtp MTP_MODEL]\n"
            "                   [--dflash DFLASH_MODEL --dflash-at P [--dflash-names REGEX] [--dflash-random]]\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_path, layers_arg, regex_arg = "", logits_mode = "f32", mtp_path, dflash_path, dflash_names;
    State st;
    int ubatch = 0, split = -1, threads = 16, dflash_at = -1;
    bool dflash_random = false;
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
        } else if (a == "-t") {
            tokens_path = next();
        } else if (a == "-o") {
            st.outdir = next();
        } else if (a == "-u") {
            ubatch = std::atoi(next().c_str());
        } else if (a == "--split") {
            split = std::atoi(next().c_str());
        } else if (a == "-l") {
            layers_arg = next();
        } else if (a == "-n") {
            regex_arg = next();
        } else if (a == "--logits") {
            logits_mode = next();
        } else if (a == "--states") {
            st.states_all = next() == "all";
        } else if (a == "--threads") {
            threads = std::atoi(next().c_str());
        } else if (a == "--mtp") {
            mtp_path = next();
        } else if (a == "--dflash") {
            dflash_path = next();
        } else if (a == "--dflash-at") {
            dflash_at = std::atoi(next().c_str());
        } else if (a == "--dflash-names") {
            dflash_names = next();
        } else if (a == "--dflash-random") {
            dflash_random = true;
        } else {
            usage();
        }
    }
    if (model_path.empty() || tokens_path.empty() || st.outdir.empty()) {
        usage();
    }
    std::vector<llama_token> tokens;
    {
        FILE * f = fopen(tokens_path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "cannot read %s\n", tokens_path.c_str());
            return 1;
        }
        int32_t x;
        while (fread(&x, 4, 1, f) == 1) {
            tokens.push_back(x);
        }
        fclose(f);
    }
    const int n = (int) tokens.size();
    if (n == 0) {
        fprintf(stderr, "no tokens\n");
        return 1;
    }
    if (split > n) {
        split = n;
    }
    const int n_prompt = split >= 0 ? split : n;
    if (ubatch <= 0) {
        ubatch = std::max(n_prompt, 1);
    }
    st.names = std::regex(regex_arg.empty() ? "$^" : regex_arg);
    for (size_t p = 0; !layers_arg.empty() && p != std::string::npos;) {
        const size_t q = layers_arg.find(',', p);
        const std::string tok = layers_arg.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok == "all") {
            st.all_layers = true;
        } else if (tok == "g") {
            st.layers.insert(-1);
        } else if (!tok.empty()) {
            st.layers.insert(std::atoi(tok.c_str()));
        }
        p = q == std::string::npos ? q : q + 1;
    }
    {  // one file per (name, layer) stays open
        struct rlimit rl;
        if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
            rl.rlim_cur = rl.rlim_max;
            setrlimit(RLIMIT_NOFILE, &rl);
        }
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
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((n + 255) / 256 * 256);
    cp.n_batch = (uint32_t) ubatch;
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    st.capture_h = !mtp_path.empty();
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    Npy logits;
    if (logits_mode != "none") {
        logits.open(st.outdir + "/logits.npy");
        logits.tail = {n_vocab};
        if (logits_mode == "f16") {
            logits.dtype = "<f2";
            logits.elem = 2;
        }
    }
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<ggml_fp16_t> half(n_vocab);
    llama_model * dfm = nullptr;
    std::vector<int32_t> tap_layers;
    std::vector<float> taps;  // [row][tap][hidden]
    const int n_embd_t = llama_model_n_embd(model);
    if (!dflash_path.empty()) {
        dfm = llama_model_load_from_file(dflash_path.c_str(), mp);
        if (!dfm || dflash_at < 0 || dflash_at >= n) {
            fprintf(stderr, "cannot load %s, or --dflash-at past the tokens\n", dflash_path.c_str());
            return 1;
        }
        tap_layers.assign(llama_model_target_layer_ids(dfm), llama_model_target_layer_ids(dfm) + llama_model_target_layer_ids_n(dfm));
        for (const int32_t l : tap_layers) {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) l, true);
        }
    }
    for (int p = 0; p < n && !dflash_random;) {
        const int nb = p < n_prompt ? std::min(ubatch, n_prompt - p) : 1;
        batch.n_tokens = nb;
        for (int i = 0; i < nb; i++) {
            batch.token[i] = tokens[p + i];
            batch.pos[i] = p + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = logits_mode != "none" || !mtp_path.empty() || p + i == n - 1;  // (--mtp: every row's h)
        }
        st.n_tokens = nb;
        st.seen.clear();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed at position %d\n", p);
            return 1;
        }
        if (logits.f) {
            for (int i = 0; i < nb; i++) {
                const float * l = llama_get_logits_ith(ctx, i);
                if (logits.elem == 2) {
                    for (int k = 0; k < n_vocab; k++) {
                        half[k] = ggml_fp32_to_fp16(l[k]);
                    }
                    logits.write(half.data(), half.size() * 2);
                } else {
                    logits.write(l, (size_t) n_vocab * 4);
                }
                logits.rows++;
            }
        }
        for (int i = 0; i < nb && dfm; i++) {
            for (const int32_t l : tap_layers) {
                const float * x = llama_get_embeddings_layer_inp(ctx, (uint32_t) l);
                if (!x) {
                    fprintf(stderr, "layer %d's input not extracted\n", l);
                    return 1;
                }
                taps.insert(taps.end(), x + (size_t) i * n_embd_t, x + (size_t) (i + 1) * n_embd_t);
            }
        }
        st.ubatch++;
        p += nb;
    }
    if (logits.f) {
        logits.close();
    }
    State sm;  // (the MTP block's run: the same ubatches)
    llama_model * mtp = nullptr;
    llama_context * cx = nullptr;
    if (!mtp_path.empty()) {
        const int n_embd = llama_model_n_embd(model);
        if (st.h.size() != (size_t) n * n_embd) {  // (no row cropped: the tensor was renamed result_norm)
            st.h.swap(st.h_result);
        }
        if (st.h.size() != (size_t) n * n_embd) {
            fprintf(stderr, "h_nextn: %zu values for %d tokens of %d\n", st.h.size(), n, n_embd);
            return 1;
        }
        llama_model_params mpm = mp;
        mpm.load_mtp = true;
        mtp = llama_model_load_from_file(mtp_path.c_str(), mpm);
        if (!mtp) {
            fprintf(stderr, "cannot load %s\n", mtp_path.c_str());
            return 1;
        }
        sm.outdir = st.outdir;
        sm.names = st.names;
        sm.layers = st.layers;
        sm.all_layers = st.all_layers;
        sm.prefix = "mtp.";
        llama_context_params cm = cp;
        cm.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cm.cb_eval_user_data = &sm;
        cx = llama_init_from_model(mtp, cm);
        if (!cx) {
            fprintf(stderr, "cannot create the MTP context\n");
            return 1;
        }
        Npy ml;
        if (logits_mode != "none") {
            ml.open(st.outdir + "/mtp.logits.npy");
            ml.tail = {n_vocab};
            if (logits_mode == "f16") {
                ml.dtype = "<f2";
                ml.elem = 2;
            }
        }
        const std::vector<float> zero(n_embd, 0.0f);
        llama_batch_ext * be = llama_batch_ext_init(cx);
        for (int p = 0; p < n;) {
            const int nb = p < n_prompt ? std::min(ubatch, n_prompt - p) : 1;
            llama_batch_ext_clear(be);
            for (int i = 0; i < nb; i++) {
                const llama_pos pos = p + i;
                const int32_t idx = llama_batch_ext_add_token(be, 0, tokens[p + i]);
                const float * h = pos ? st.h.data() + (size_t) (pos - 1) * n_embd : zero.data();
                if (idx < 0 || !llama_batch_ext_set_pos(be, idx, &pos) ||
                    !llama_batch_ext_set_embd_state(be, idx, {h, 1, (size_t) n_embd}) ||
                    !llama_batch_ext_set_output_logits(be, idx, logits_mode != "none" || pos == n - 1)) {
                    fprintf(stderr, "MTP batch at position %d\n", pos);
                    return 1;
                }
            }
            sm.n_tokens = nb;
            sm.seen.clear();
            if (llama_process(cx, LLAMA_PROCESS_TYPE_DECODE, be) != 0) {
                fprintf(stderr, "llama_process (MTP) failed at position %d\n", p);
                return 1;
            }
            if (ml.f) {
                for (int i = 0; i < nb; i++) {
                    const float * l = llama_get_logits_ith(cx, i);
                    if (ml.elem == 2) {
                        for (int k = 0; k < n_vocab; k++) {
                            half[k] = ggml_fp32_to_fp16(l[k]);
                        }
                        ml.write(half.data(), half.size() * 2);
                    } else {
                        ml.write(l, (size_t) n_vocab * 4);
                    }
                    ml.rows++;
                }
            }
            sm.ubatch++;
            p += nb;
        }
        llama_batch_ext_free(be);
        if (ml.f) {
            ml.close();
        }
    }
    State sd;  // (the drafter's run)
    llama_context * cd = nullptr;
    if (dfm) {
        const int P = dflash_at, nt = (int) tap_layers.size(), row = nt * n_embd_t;
        if (dflash_random) {
            std::mt19937 gen(1);
            std::normal_distribution<float> normal(0.0f, 1.0f / 16);
            taps.resize((size_t) P * row);
            for (float & v : taps) {
                v = normal(gen);
            }
        }
        sd.outdir = st.outdir;
        sd.prefix = "dflash.";
        sd.names = std::regex(dflash_names.empty() ? "$^" : dflash_names);
        sd.all_layers = true;
        sd.layers = {-1};
        char buf[32] = {};
        llama_model_meta_val_str(dfm, "dflash.block_size", buf, sizeof(buf));
        const int block = std::atoi(buf);
        llama_context_params cf = cp;
        cf.ctx_other = ctx;
        cf.n_ctx = (uint32_t) ((P + block + 255) / 256 * 256);
        cf.n_batch = cf.n_ubatch = (uint32_t) std::max(ubatch, block);
        cf.n_outputs_max = (uint32_t) block;
        cf.cb_eval_user_data = &sd;
        cd = llama_init_from_model(dfm, cf);
        if (!cd) {
            fprintf(stderr, "cannot create the drafter's context\n");
            return 1;
        }
        llama_set_embeddings_nextn(cd, true, false);
        llama_set_causal_attn(cd, false);
        Npy tf;
        tf.open(st.outdir + "/dflash.taps.npy");
        tf.tail = {nt, n_embd_t};
        tf.rows = P;
        tf.write(taps.data(), (size_t) P * row * 4);
        tf.close();
        llama_batch_ext * be = llama_batch_ext_init(cd);
        for (int p = 0; p < P;) {
            const int nb = std::min((int) cf.n_ubatch, P - p);
            llama_batch_ext_clear(be);
            for (int i = 0; i < nb; i++) {
                const llama_pos pos[4] = {p + i, p + i, p + i, 0};
                const int32_t idx = llama_batch_ext_add_embd(be, 0, {taps.data() + (size_t) (p + i) * row, 1, (size_t) row});
                if (idx < 0 || !llama_batch_ext_set_pos(be, idx, pos)) {
                    fprintf(stderr, "injection batch at position %d\n", p + i);
                    return 1;
                }
            }
            sd.n_tokens = nb;
            sd.seen.clear();
            if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, be) != 0) {
                fprintf(stderr, "llama_process (injection) failed at position %d\n", p);
                return 1;
            }
            sd.ubatch++;
            p += nb;
        }
        llama_batch_ext_clear(be);
        const llama_token mask = llama_vocab_mask(llama_model_get_vocab(dfm));
        for (int i = 0; i < block; i++) {
            const llama_pos pos = P + i;
            const int32_t idx = llama_batch_ext_add_token(be, 0, i == 0 ? tokens[P] : mask);
            if (idx < 0 || !llama_batch_ext_set_pos(be, idx, &pos)) {
                fprintf(stderr, "block batch\n");
                return 1;
            }
        }
        sd.n_tokens = block;
        sd.seen.clear();
        if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, be) != 0) {
            fprintf(stderr, "llama_process (block) failed\n");
            return 1;
        }
        llama_batch_ext_free(be);
        Npy lf;
        lf.open(st.outdir + "/dflash.lattice.npy");
        lf.tail = {n_embd_t};
        lf.rows = block;
        lf.write(llama_get_embeddings_nextn(cd), (size_t) block * n_embd_t * 4);
        lf.close();
    }
    for (State * s : {&st, &sm, &sd}) {
        for (auto & [name, f] : s->files) {
            f.close();
        }
        for (auto & [name, sv] : s->last) {
            Npy f;
            f.open(s->outdir + "/" + s->prefix + name + ".npy");
            f.tail = {sv.first.begin() + 1, sv.first.end()};
            f.rows = sv.first[0];
            f.write(sv.second.data(), sv.second.size() * 4);
            f.close();
        }
    }
    {
        FILE * f = fopen((st.outdir + "/meta.json").c_str(), "w");
        fprintf(f, "{\"model\": \"%s\", \"tokens\": %d, \"ubatch\": %d, \"split\": %d, \"ubatches\": %d, \"logits\": \"%s\", "
                   "\"mtp\": \"%s\"}\n",
                model_path.c_str(), n, ubatch, split, st.ubatch, logits_mode.c_str(), mtp_path.c_str());
        fclose(f);
    }
    llama_batch_free(batch);
    if (cx) {
        llama_free(cx);
        llama_model_free(mtp);
    }
    if (dfm) {
        if (cd) {
            llama_free(cd);
        }
        llama_model_free(dfm);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
