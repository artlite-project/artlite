// Goldens of llama.cpp's LFM3-Audio speech output (libmtmd's lfm3a_gen: the depthformer and the codec decoder of the
// mmproj) on the CPU backend, as .npy files.
//
// Built against a prebuilt llama.cpp tree (include/, ggml/include/, tools/mtmd/, build/bin/libmtmd.so) by
// gen_ref.py.
//
//   lfm3a_gen_dump -m MODEL --mmproj MMPROJ -o OUTDIR [-p PROMPT] [--h H.f32] [--passes P.i32] [--feed F.i32]
//                  [--frames W.i32 --chunk K] [--gen N --seed S] [-n REGEX] [--threads N]
//
// - PROMPT: a text file (special tokens parsed, no BOS added): the language model runs it (embeddings on) -> h.npy:
//   the hidden state (n_embd) after its last token (llama_get_embeddings_ith: the output norm's output);
// - H: raw float32 (n_embd): the hidden state the passes take instead of the prompt's;
// - P: raw int32 (n, n_codebooks): per row a depthformer pass on the hidden state with those codes revealed (-1 masked)
//   -> logits-<i>.npy (n_codebooks, n_codes); a row with every code known: its feedback row fb-<i>.npy (n_embd);
// - F: raw int32 (n, n_codebooks): teacher-forced frames after the prompt: each frame's feedback row decoded at the next
//   position -> hf-<i>.npy, the hidden state after frame i, and before that, with the prompt's hidden state for the
//   first, logits-f<i>.npy: the frame's first pass (every code masked);
// - --gen N: up to N frames after the prompt sampled as llama.cpp's lfm3a_gen_audio_pipeline samples them (a std::mt19937
//   of seed S: per frame a shuffled codebook order, 3 passes revealing 4 codebooks each, the mmproj's sampling) ->
//   codes.npy (frames, n_codebooks; the end-of-audio frame not included), hg-<i>.npy (the hidden state frame i's
//   passes took);
// - W: raw int32 (n, n_codebooks): frames through the codec decoder and the ISTFT in calls of K frames, then the flush
//   -> pcm.npy (float32 samples), pcm_calls.npy (int32: the samples of each call, the flush's last);
// - REGEX: the generator graphs' tensors to dump (names without the "-<il>" suffix): <name>[-<il>].g<k>.npy, k the
//   graph's index in this run (passes, then feedback rows, then codec calls, in the order they ran).
//
// Backend: CPU only, F32 KV cache, flash attention off.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

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
    const size_t total = head.size() + 2 + dict.size() + 1;
    dict += std::string((64 - total % 64) % 64, ' ') + "\n";
    const uint16_t n = (uint16_t) dict.size();
    head += (char) (n & 0xFF);
    head += (char) (n >> 8);
    head += dict;
    fwrite(head.data(), 1, head.size(), f);
    fwrite(data, 1, bytes, f);
    fclose(f);
}

static void save_f32(const std::string & path, const float * x, std::vector<int64_t> shape) {
    size_t n = 1;
    for (int64_t s : shape) {
        n *= (size_t) s;
    }
    write_npy(path, x, n * 4, shape, "<f4");
}

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        exit(1);
    }
    std::string s;
    char buf[4096];
    size_t k;
    while ((k = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, k);
    }
    fclose(f);
    return s;
}

template <typename T> static std::vector<T> read_raw(const std::string & path) {
    const std::string s = read_file(path);
    std::vector<T> v(s.size() / sizeof(T));
    memcpy(v.data(), s.data(), v.size() * sizeof(T));
    return v;
}

struct State {
    std::string outdir;
    std::regex names;
    bool any = false;
    int graph = -1;                   // (the generator graph: counted at its first node)
    std::map<std::string, int> seen;  // (names dumped in the current graph)
};

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    State & st = *(State *) user;
    if (!st.any || strpbrk(t->name, " (") || t->type != GGML_TYPE_F32) {
        return false;
    }
    std::string s(t->name);
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        s.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        s = s.substr(0, dash);
    }
    if (!std::regex_match(s, st.names)) {
        return false;
    }
    if (ask) {
        return true;
    }
    std::vector<float> f;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                const char * row = (const char *) t->data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1];
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    f.push_back(*(const float *) (row + i0 * t->nb[0]));
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
    std::string name = t->name;
    const int dup = st.seen[name]++;
    if (dup) {
        name += "." + std::to_string(dup);
    }
    save_f32(st.outdir + "/" + name + ".g" + std::to_string(st.graph) + ".npy", f.data(), shape);
    return true;
}

static void usage() {
    fprintf(stderr, "usage: lfm3a_gen_dump -m MODEL --mmproj MMPROJ -o OUTDIR [-p PROMPT] [--h H.f32] [--passes P.i32]\n"
                    "                      [--feed F.i32] [--frames W.i32 --chunk K] [--gen N --seed S] [-n REGEX]\n"
                    "                      [--threads N]\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, mmproj_path, prompt_path, h_path, passes_path, feed_path, frames_path, regex;
    State st;
    int threads = 16, chunk = 4, gen = 0;
    unsigned seed = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            usage();
        }
        const std::string v = argv[++i];
        if (a == "-m") model_path = v;
        else if (a == "--mmproj") mmproj_path = v;
        else if (a == "-o") st.outdir = v;
        else if (a == "-p") prompt_path = v;
        else if (a == "--h") h_path = v;
        else if (a == "--passes") passes_path = v;
        else if (a == "--feed") feed_path = v;
        else if (a == "--frames") frames_path = v;
        else if (a == "--chunk") chunk = atoi(v.c_str());
        else if (a == "-n") regex = v;
        else if (a == "--gen") gen = atoi(v.c_str());
        else if (a == "--seed") seed = (unsigned) strtoul(v.c_str(), nullptr, 10);
        else if (a == "--threads") threads = atoi(v.c_str());
        else usage();
    }
    if (model_path.empty() || mmproj_path.empty() || st.outdir.empty()) {
        usage();
    }
    st.any = !regex.empty();
    st.names = std::regex(st.any ? regex : "^$");

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
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = false;
    vp.n_threads = threads;
    vp.warmup = false;
    vp.print_timings = false;
    vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vp.cb_eval = eval_cb;
    vp.cb_eval_user_data = &st;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, vp);
    if (!mctx || mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_LFM3A) {
        fprintf(stderr, "%s: no LFM3-Audio generator\n", mmproj_path.c_str());
        return 1;
    }
    mtmd_gen_state_params sp = mtmd_gen_state_params_default();
    sp.seed = 0;
    mtmd_gen_state * gst = mtmd_gen_state_init(mctx, sp);
    const int n_cb = 12;

    std::vector<float> h;
    llama_context * lctx = nullptr;
    int pos = 0;
    if (!prompt_path.empty()) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_batch = 2048;
        cp.n_ubatch = 2048;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.no_perf = true;
        cp.embeddings = true;
        lctx = llama_init_from_model(model, cp);
        const std::string text = read_file(prompt_path);
        std::vector<llama_token> ids(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false,
                                     true);
        ids.resize((size_t) n);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            b.token[i] = ids[(size_t) i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = i == n - 1;
        }
        b.n_tokens = n;
        if (llama_decode(lctx, b) != 0) {
            fprintf(stderr, "prompt decode failed\n");
            return 1;
        }
        llama_batch_free(b);
        pos = n;
        const float * he = llama_get_embeddings_ith(lctx, -1);
        h.assign(he, he + n_embd);
        save_f32(st.outdir + "/h.npy", h.data(), {n_embd});
        std::vector<int32_t> idv(ids.begin(), ids.end());
        write_npy(st.outdir + "/prompt_ids.npy", idv.data(), idv.size() * 4, {(int64_t) idv.size()}, "<i4");
    }
    if (!h_path.empty()) {
        h = read_raw<float>(h_path);
    }

    auto pass = [&](std::vector<int32_t> codes, const std::vector<float> & hs, const std::string & tag) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd = const_cast<float *>(hs.data());
        inp.codes = codes.data();
        inp.n_codes = codes.size();
        mtmd_gen_out out{};
        st.graph++;
        st.seen.clear();
        if (mtmd_gen_audio_process(mctx, gst, &inp, &out) != 0) {
            fprintf(stderr, "depthformer pass failed\n");
            exit(1);
        }
        bool known = true;
        for (int32_t c : codes) {
            known = known && c >= 0;
        }
        if (known) {
            save_f32(st.outdir + "/fb-" + tag + ".npy", out.embd, {n_embd});
            return std::vector<float>(out.embd, out.embd + n_embd);
        }
        save_f32(st.outdir + "/logits-" + tag + ".npy", out.feats, {n_cb, (int64_t) out.n_feats / n_cb});
        return std::vector<float>();
    };

    if (!passes_path.empty()) {
        const std::vector<int32_t> p = read_raw<int32_t>(passes_path);
        for (size_t i = 0; i < p.size() / n_cb; i++) {
            pass(std::vector<int32_t>(p.begin() + i * n_cb, p.begin() + (i + 1) * n_cb), h, std::to_string(i));
        }
    }

    auto decode_row = [&](const std::vector<float> & fb) {
        llama_batch b = llama_batch_init(1, n_embd, 1);
        memcpy(b.embd, fb.data(), (size_t) n_embd * 4);
        b.pos[0] = pos++;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;
        if (llama_decode(lctx, b) != 0) {
            fprintf(stderr, "frame decode failed\n");
            exit(1);
        }
        llama_batch_free(b);
        const float * he = llama_get_embeddings_ith(lctx, -1);
        return std::vector<float>(he, he + n_embd);
    };

    if (gen > 0) {  // (lfm3a_gen_audio_pipeline::sample_frame, apply_rep_penalty and sample)
        if (!lctx) {
            fprintf(stderr, "--gen needs a prompt\n");
            return 1;
        }
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        std::mt19937 rng(seed);
        std::vector<int32_t> hist;
        std::vector<float> hs = h;
        for (int f = 0; f < gen; f++) {
            save_f32(st.outdir + "/hg-" + std::to_string(f) + ".npy", hs.data(), {n_embd});
            std::vector<int> order(n_cb);
            for (int i = 0; i < n_cb; i++) {
                order[(size_t) i] = i;
            }
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<int32_t> codes(n_cb, -1);
            int k = 0;
            size_t n_vocab = 0;
            for (int p = 0; p < 3; p++) {
                mtmd_gen_inp inp = def;
                inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
                inp.embd = hs.data();
                inp.codes = codes.data();
                inp.n_codes = codes.size();
                mtmd_gen_out out{};
                st.graph++;
                st.seen.clear();
                if (mtmd_gen_audio_process(mctx, gst, &inp, &out) != 0) {
                    fprintf(stderr, "depthformer pass failed\n");
                    return 1;
                }
                n_vocab = out.n_feats / n_cb;
                std::vector<float> lg(out.feats, out.feats + out.n_feats);
                for (int j = 0; j < 4; j++, k++) {
                    const int i = order[(size_t) k];
                    float * l = lg.data() + (size_t) i * n_vocab;
                    const size_t n_hist = hist.size() / n_cb;
                    const size_t n_win = std::min(n_hist, (size_t) std::max(def.penalty_last_n, 0));
                    if (def.penalty_repeat > 1.0f && def.penalty_last_n > 0) {
                        std::vector<bool> seen(n_vocab, false);
                        for (size_t fr = n_hist - n_win; fr < n_hist; fr++) {
                            const int32_t c = hist[fr * n_cb + (size_t) i];
                            if (!seen[(size_t) c]) {
                                seen[(size_t) c] = true;
                                l[c] = l[c] > 0.0f ? l[c] / def.penalty_repeat : l[c] * def.penalty_repeat;
                            }
                        }
                    }
                    float lmin = -INFINITY;
                    if (def.top_k > 0 && (size_t) def.top_k < n_vocab) {
                        std::vector<float> sorted(l, l + n_vocab);
                        std::nth_element(sorted.begin(), sorted.begin() + (def.top_k - 1), sorted.end(), std::greater<float>());
                        lmin = sorted[(size_t) def.top_k - 1];
                    }
                    const float lmax = *std::max_element(l, l + n_vocab);
                    std::vector<double> prob(n_vocab);
                    double sum = 0.0;
                    for (size_t v = 0; v < n_vocab; v++) {
                        prob[v] = l[v] < lmin ? 0.0 : std::exp((double) (l[v] - lmax) / def.temp);
                        sum += prob[v];
                    }
                    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
                    int32_t pick = (int32_t) n_vocab - 1;
                    for (size_t v = 0; v < n_vocab; v++) {
                        r -= prob[v];
                        if (r <= 0.0) {
                            pick = (int32_t) v;
                            break;
                        }
                    }
                    codes[(size_t) i] = pick;
                }
            }
            if (std::find(codes.begin(), codes.end(), (int32_t) n_vocab - 1) != codes.end()) {
                break;
            }
            hist.insert(hist.end(), codes.begin(), codes.end());
            hs = decode_row(pass(codes, hs, "g" + std::to_string(f)));
        }
        write_npy(st.outdir + "/codes.npy", hist.data(), hist.size() * 4, {(int64_t) hist.size() / n_cb, n_cb}, "<i4");
    }

    if (!feed_path.empty()) {
        if (!lctx) {
            fprintf(stderr, "--feed needs a prompt\n");
            return 1;
        }
        const std::vector<int32_t> fr = read_raw<int32_t>(feed_path);
        std::vector<float> hs = h;
        for (size_t i = 0; i < fr.size() / n_cb; i++) {
            pass(std::vector<int32_t>(n_cb, -1), hs, "f" + std::to_string(i));
            hs = decode_row(pass(std::vector<int32_t>(fr.begin() + i * n_cb, fr.begin() + (i + 1) * n_cb), hs,
                                 "f" + std::to_string(i)));
            save_f32(st.outdir + "/hf-" + std::to_string(i) + ".npy", hs.data(), {n_embd});
        }
    }

    if (!frames_path.empty()) {
        std::vector<int32_t> w = read_raw<int32_t>(frames_path);
        const size_t nf = w.size() / n_cb;
        std::vector<float> pcm;
        std::vector<int32_t> calls;
        for (size_t f0 = 0;; f0 += (size_t) chunk) {
            const size_t k = f0 < nf ? std::min((size_t) chunk, nf - f0) : 0;  // (the last call: the flush)
            mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
            inp.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            inp.codes = k ? w.data() + f0 * n_cb : nullptr;
            inp.n_codes = k * n_cb;
            mtmd_gen_out out{};
            st.graph++;
            st.seen.clear();
            if (mtmd_gen_audio_process(mctx, gst, &inp, &out) != 0) {
                fprintf(stderr, "codec call failed\n");
                return 1;
            }
            pcm.insert(pcm.end(), out.audio, out.audio + out.n_samples);
            calls.push_back((int32_t) out.n_samples);
            if (!k) {
                break;
            }
        }
        save_f32(st.outdir + "/pcm.npy", pcm.data(), {(int64_t) pcm.size()});
        write_npy(st.outdir + "/pcm_calls.npy", calls.data(), calls.size() * 4, {(int64_t) calls.size()}, "<i4");
    }

    mtmd_gen_state_free(gst);
    mtmd_free(mctx);
    if (lctx) {
        llama_free(lctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
