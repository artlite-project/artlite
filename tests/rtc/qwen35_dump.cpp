// Golden tensors of a llama.cpp model run (Qwen3.5 / 3.6 "qwen35" GGUF files): teacher forcing of an exact
// token sequence on the CPU backend, with the intermediate tensors of selected layers dumped as .npy files.
//
// Built against a prebuilt llama.cpp tree (include/, ggml/include/, build/bin/libllama.so) by qwen35_ref.py.
//
//   qwen35_dump -m MODEL -t TOKENS -o OUTDIR [-u UBATCH] [--split P] [-l LAYERS] [-n REGEX]
//               [--logits f32|f16|none] [--states last|all] [--threads N]
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
// Backend: CPU only (no devices), F32 KV cache, flash attention off.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <sys/resource.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    State & st = *(State *) user;
    if (ask) {
        return wanted(st, t);
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
                f.open(st.outdir + "/" + name + ".npy");
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
        f.open(st.outdir + "/" + name + ".npy");
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
            "                   [--logits f32|f16|none] [--states last|all] [--threads N]\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_path, layers_arg, regex_arg = "", logits_mode = "f32";
    State st;
    int ubatch = 0, split = -1, threads = 16;
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
    for (int p = 0; p < n;) {
        const int nb = p < n_prompt ? std::min(ubatch, n_prompt - p) : 1;
        batch.n_tokens = nb;
        for (int i = 0; i < nb; i++) {
            batch.token[i] = tokens[p + i];
            batch.pos[i] = p + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = logits_mode != "none" || p + i == n - 1;
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
        st.ubatch++;
        p += nb;
    }
    if (logits.f) {
        logits.close();
    }
    for (auto & [name, f] : st.files) {
        f.close();
    }
    for (auto & [name, sv] : st.last) {
        Npy f;
        f.open(st.outdir + "/" + name + ".npy");
        f.tail = {sv.first.begin() + 1, sv.first.end()};
        f.rows = sv.first[0];
        f.write(sv.second.data(), sv.second.size() * 4);
        f.close();
    }
    {
        FILE * f = fopen((st.outdir + "/meta.json").c_str(), "w");
        fprintf(f, "{\"model\": \"%s\", \"tokens\": %d, \"ubatch\": %d, \"split\": %d, \"ubatches\": %d, \"logits\": \"%s\"}\n",
                model_path.c_str(), n, ubatch, split, st.ubatch, logits_mode.c_str());
        fclose(f);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
