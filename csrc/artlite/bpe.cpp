#include "artlite/bpe.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "artlite/error.h"

namespace artlite {

namespace {

using Pieces = std::vector<std::pair<size_t, size_t>>;  // (first codepoint, length)

// One regex over every piece: its matches and the gaps between them (std::regex_iterator over the piece, as
// unicode_regex_split_stl). match(i, end) is the length of the regex's match at i inside [i, end), 0 if none.
template <class Match>
Pieces split(const Pieces& in, Match match) {
  Pieces out;
  out.reserve(in.size() * 2);
  for (const auto& [b, len] : in) {
    const size_t e = b + len;
    size_t gap = b, i = b;
    while (i < e) {
      const size_t m = match(i, e);
      if (!m) {
        i++;
        continue;
      }
      if (i > gap) out.emplace_back(gap, i - gap);
      out.emplace_back(i, m);
      i += m;
      gap = i;
    }
    if (gap < e) out.emplace_back(gap, e - gap);
  }
  return out;
}

// The end of a run of pred from i, at most e.
template <class Pred>
size_t run(size_t i, size_t e, Pred pred) {
  while (i < e && pred(i)) i++;
  return i;
}

// The first two regexes of spark2_5's and flint's pre-tokenizers: \p{N}{1,3}, then [一-龥぀-ゟ゠-ヿ]+ (std::wregex on the
// codepoints).
Pieces digits_cjk(const uint32_t* cp, const uint8_t* cls, size_t n) {
  auto num = [&](size_t i) { return cls[i] == PC_N; };
  Pieces p{{0, n}};
  p = split(p, [&](size_t i, size_t e) { return num(i) ? std::min<size_t>(run(i, e, num) - i, 3) : 0; });
  auto cjk = [&](size_t i) {
    const uint32_t c = cp[i];
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
  };
  return split(p, [&](size_t i, size_t e) { return run(i, e, cjk) - i; });
}

std::vector<uint32_t> lengths(const Pieces& p) {
  std::vector<uint32_t> out(p.size());
  for (size_t k = 0; k < p.size(); k++) out[k] = uint32_t(p[k].second);
  return out;
}

}  // namespace

std::vector<uint32_t> pretokenize_spark2_5(const uint32_t* cp, const uint8_t* cls, size_t n) {
  if (!n) return {};
  auto is = [&](size_t i, PreClass c) { return cls[i] == c; };
  auto ws = [&](size_t i) { return cls[i] == PC_WS; };
  auto lm = [&](size_t i) { return cls[i] == PC_L || cls[i] == PC_M; };
  auto ps = [&](size_t i) { return cls[i] == PC_P || cls[i] == PC_S; };
  auto ascii_letter = [&](size_t i) { return (cp[i] | 0x20) >= 'a' && (cp[i] | 0x20) <= 'z'; };
  auto ascii_punct = [&](size_t i) {  // [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]
    const uint32_t c = cp[i];
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
  };

  Pieces p = digits_cjk(cp, cls, n);
  // the main split: its alternatives in order, the first that matches at i
  p = split(p, [&](size_t i, size_t e) -> size_t {
    if (ascii_punct(i) && i + 1 < e && ascii_letter(i + 1)) return run(i + 1, e, ascii_letter) - i;
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ (the optional character taken when letters follow it)
    const bool pre = cp[i] != '\r' && cp[i] != '\n' && !is(i, PC_L) && !ps(i);
    if (pre && i + 1 < e && lm(i + 1)) return run(i + 1, e, lm) - i;
    if (lm(i)) return run(i, e, lm) - i;
    // ?[\p{P}\p{S}]+
    if (cp[i] == ' ' && i + 1 < e && ps(i + 1)) return run(i + 1, e, ps) - i;
    if (ps(i)) return run(i, e, ps) - i;
    if (cp[i] == '\r' || cp[i] == '\n') return 1;
    if (ws(i)) {  // \s+(?!\S) (a run followed by a non-space gives it its last), else \s+
      const size_t j = run(i, e, ws);
      return j == e || j - i == 1 ? j - i : j - i - 1;
    }
    return 0;
  });
  // \p{N}
  p = split(p, [&](size_t i, size_t) -> size_t { return is(i, PC_N) ? 1 : 0; });
  return lengths(p);
}

std::vector<uint32_t> pretokenize_flint(const uint32_t* cp, const uint8_t* cls, size_t n) {
  if (!n) return {};
  auto lm = [&](size_t i) { return cls[i] == PC_L || cls[i] == PC_M; };
  auto ps = [&](size_t i) { return cls[i] == PC_P || cls[i] == PC_S; };
  auto ws = [&](size_t i) { return cls[i] == PC_WS; };
  auto nl = [&](size_t i) { return cp[i] == '\r' || cp[i] == '\n'; };
  // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?: its length at i
  auto contraction = [&](size_t i, size_t e) -> size_t {
    if (i + 1 >= e || cp[i] != '\'') return 0;
    auto lower = [&](size_t k) { return k < e && cp[k] < 0x80 ? char(cp[k] | 0x20) : '\0'; };
    const char a = lower(i + 1), b = lower(i + 2);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' || a == 'v') && b == 'e') return 3;
    if (a == 'l' && b == 'l') return 3;
    return 0;
  };

  Pieces p = digits_cjk(cp, cls, n);
  // the o200k-style split as llama.cpp runs it (std::regex on the categories, case classes collapsed to \p{L}):
  // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(contraction)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  p = split(p, [&](size_t i, size_t e) -> size_t {
    const bool pre = !nl(i) && cls[i] != PC_L && cls[i] != PC_N;
    if ((pre && i + 1 < e && lm(i + 1)) || lm(i)) {
      const size_t j = run(lm(i) ? i : i + 1, e, lm);
      return j - i + contraction(j, e);
    }
    if (ps(i) || (cp[i] == ' ' && i + 1 < e && ps(i + 1))) {
      const size_t j = run(run(i + !ps(i), e, ps), e, nl);
      return j - i;
    }
    if (ws(i)) {
      const size_t j = run(i, e, ws);
      for (size_t k = j; k > i; k--)  // \s*[\r\n]+: to the run's last line break
        if (nl(k - 1)) return k - i;
      return j == e || j - i == 1 ? j - i : j - i - 1;  // \s+(?!\S), else \s+
    }
    return 0;
  });
  return lengths(p);
}

std::vector<uint32_t> pretokenize_gpt4o(const uint32_t* cp, const uint8_t* cls, size_t n) {
  if (!n) return {};
  auto num = [&](size_t i) { return cls[i] == PC_N; };
  auto ws = [&](size_t i) { return cls[i] == PC_WS; };
  auto nl = [&](size_t i) { return cp[i] == '\r' || cp[i] == '\n'; };
  auto lm = [&](size_t i) { return cls[i] == PC_L || cls[i] == PC_M; };
  // (?=[\p{L}\p{M}])([^a-z]) and (?=[\p{L}\p{M}])([^A-Z]): a letter or mark that is not ASCII lowercase / uppercase
  auto up = [&](size_t i) { return lm(i) && !(cp[i] >= 'a' && cp[i] <= 'z'); };
  auto lo = [&](size_t i) { return lm(i) && !(cp[i] >= 'A' && cp[i] <= 'Z'); };
  // [^\r\n\p{L}\p{N}] (a mark too: \p{M} is no \p{L}) and [^\s\p{L}\p{N}]
  auto pre = [&](size_t i) { return !nl(i) && cls[i] != PC_L && !num(i); };
  auto other = [&](size_t i) { return !ws(i) && cls[i] != PC_L && !num(i); };
  auto contraction = [&](size_t i, size_t e) -> size_t {  // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
    if (i + 1 >= e || cp[i] != '\'') return 0;
    auto lower = [&](size_t k) { return k < e && cp[k] < 0x80 ? char(cp[k] | 0x20) : '\0'; };
    const char a = lower(i + 1), b = lower(i + 2);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' || a == 'v') && b == 'e') return 3;
    if (a == 'l' && b == 'l') return 3;
    return 0;
  };
  // up* lo+ from j, std::regex's backtracking: the longest run of up whose next character starts a run of lo, else
  // shorter; the end of that lo run, 0 if none
  auto upper_lower = [&](size_t j, size_t e) -> size_t {
    const size_t m = run(j, e, up);
    for (size_t u = m;; u--) {
      if (u < e && lo(u)) return run(u, e, lo);
      if (u == j) return 0;
    }
  };
  Pieces p{{0, n}};
  p = split(p, [&](size_t i, size_t e) -> size_t {
    for (const size_t j : {i + 1, i}) {  // up* lo+, with the optional character, then without it
      if (j == i + 1 && !(pre(i) && j < e)) continue;
      if (const size_t k = upper_lower(j, e)) return k - i + contraction(k, e);
    }
    for (const size_t j : {i + 1, i}) {  // up+ lo*
      if (j == i + 1 && !(pre(i) && j < e)) continue;
      if (j < e && up(j)) {
        const size_t k = run(run(j, e, up), e, lo);
        return k - i + contraction(k, e);
      }
    }
    if (num(i)) return std::min<size_t>(run(i, e, num) - i, 3);
    if (other(i) || (cp[i] == ' ' && i + 1 < e && other(i + 1))) {
      const size_t j = run(i + (cp[i] == ' ' ? 1 : 0), e, other);
      return run(j, e, [&](size_t k) { return nl(k) || cp[k] == '/'; }) - i;
    }
    if (ws(i)) {
      const size_t j = run(i, e, ws);
      for (size_t k = j; k > i; k--)  // \s*[\r\n]+: to the run's last line break
        if (nl(k - 1)) return k - i;
      return j == e || j - i == 1 ? j - i : j - i - 1;  // \s+(?!\S), else \s+
    }
    return 0;
  });
  return lengths(p);
}

namespace {

// GPT-2's bytes_to_unicode: a byte's character in the vocabulary's strings, as UTF-8
std::string byte_char(int b) {
  uint32_t c = b;
  if (!((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))) {
    int k = 0;  // (the bytes outside those ranges, in order, from U+0100)
    for (int x = 0; x < b; x++)
      if (!((x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF))) k++;
    c = 0x100 + k;
  }
  std::string s;
  if (c < 0x80) {
    s += char(c);
  } else {
    s += char(0xC0 | (c >> 6));
    s += char(0x80 | (c & 0x3F));
  }
  return s;
}

void utf8(uint32_t c, std::vector<uint8_t>& out) {
  if (c < 0x80) {
    out.push_back(uint8_t(c));
  } else if (c < 0x800) {
    out.push_back(uint8_t(0xC0 | (c >> 6)));
    out.push_back(uint8_t(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(uint8_t(0xE0 | (c >> 12)));
    out.push_back(uint8_t(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(uint8_t(0x80 | (c & 0x3F)));
  } else {
    out.push_back(uint8_t(0xF0 | (c >> 18)));
    out.push_back(uint8_t(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(uint8_t(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(uint8_t(0x80 | (c & 0x3F)));
  }
}

}  // namespace

BpeEncoder::BpeEncoder(const std::vector<std::string>& tokens, const std::vector<std::string>& merges,
                       bool ignore_merges) {
  std::unordered_map<std::string, int32_t> id;
  id.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) id.emplace(tokens[i], int32_t(i));
  for (int b = 0; b < 256; b++) {
    byte_char_[b] = byte_char(b);
    auto it = id.find(byte_char_[b]);  // (bytes no UTF-8 text holds, 0xC0, 0xC1, 0xF5 ..: maybe none)
    byte_id_[b] = it == id.end() ? -1 : it->second;
  }
  merges_.reserve(merges.size());
  for (size_t r = 0; r < merges.size(); r++) {
    const std::string& m = merges[r];
    const size_t sp = m.find(' ', 1);  // (a part may be a space's character, never a space)
    if (sp == std::string::npos) throw Error("BpeEncoder: merge '" + m + "' has no two parts");
    auto a = id.find(m.substr(0, sp)), b = id.find(m.substr(sp + 1)), ab = id.find(m.substr(0, sp) + m.substr(sp + 1));
    if (a == id.end() || b == id.end() || ab == id.end()) throw Error("BpeEncoder: merge '" + m + "' of non-tokens");
    merges_.emplace(uint64_t(uint32_t(a->second)) << 32 | uint32_t(b->second), uint64_t(r) << 32 | uint32_t(ab->second));
  }
  if (ignore_merges) words_ = std::move(id);
}

std::vector<int32_t> BpeEncoder::encode(const uint32_t* cp, size_t n, const uint32_t* lens, size_t n_pieces) const {
  struct Sym {
    int32_t id, prev, next;
  };
  struct Pair {
    uint32_t rank;
    int32_t left, right, lid, rid;
    bool operator<(const Pair& o) const {  // (the queue's top: the lowest rank, then the leftmost)
      return rank != o.rank ? rank > o.rank : left > o.left;
    }
  };
  std::vector<int32_t> out;
  out.reserve(n);
  std::vector<uint8_t> bytes;
  std::vector<Sym> sym;
  size_t at = 0;
  for (size_t k = 0; k < n_pieces; k++) {
    if (at + lens[k] > n) throw Error("BpeEncoder.encode: pieces past the text");
    bytes.clear();
    for (size_t i = at; i < at + lens[k]; i++) utf8(cp[i], bytes);
    at += lens[k];
    if (!words_.empty() && bytes.size() > 1) {  // (a piece that is a token: that token)
      std::string word;
      for (const uint8_t b : bytes) word += byte_char_[b];
      if (auto it = words_.find(word); it != words_.end()) {
        out.push_back(it->second);
        continue;
      }
    }
    sym.resize(bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) {
      if (byte_id_[bytes[i]] < 0) throw Error("BpeEncoder.encode: byte " + std::to_string(bytes[i]) + " is not a token");
      sym[i] = {byte_id_[bytes[i]], int32_t(i) - 1, i + 1 < bytes.size() ? int32_t(i + 1) : -1};
    }
    std::priority_queue<Pair> q;
    auto add = [&](int32_t l, int32_t r) {
      if (l < 0 || r < 0) return;
      auto it = merges_.find(uint64_t(uint32_t(sym[l].id)) << 32 | uint32_t(sym[r].id));
      if (it != merges_.end()) q.push({uint32_t(it->second >> 32), l, r, sym[l].id, sym[r].id});
    };
    for (size_t i = 1; i < bytes.size(); i++) add(int32_t(i - 1), int32_t(i));
    while (!q.empty()) {
      const Pair p = q.top();
      q.pop();
      Sym &l = sym[p.left], &r = sym[p.right];
      if (l.id != p.lid || r.id != p.rid || l.next != p.right || r.id < 0) continue;  // (a stale pair)
      l.id = int32_t(uint32_t(merges_.at(uint64_t(uint32_t(p.lid)) << 32 | uint32_t(p.rid))));
      r.id = -1;
      l.next = r.next;
      if (r.next >= 0) sym[r.next].prev = p.left;
      add(l.prev, p.left);
      add(p.left, l.next);
    }
    for (int32_t i = bytes.empty() ? -1 : 0; i >= 0; i = sym[i].next) out.push_back(sym[i].id);
  }
  return out;
}

}  // namespace artlite
