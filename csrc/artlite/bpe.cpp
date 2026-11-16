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
  auto run = [&](size_t i, size_t e, auto pred) {  // the end of a run of pred from i
    while (i < e && pred(i)) i++;
    return i;
  };

  Pieces p{{0, n}};
  // \p{N}{1,3}
  p = split(p, [&](size_t i, size_t e) { return is(i, PC_N) ? std::min<size_t>(run(i, e, [&](size_t j) { return is(j, PC_N); }) - i, 3) : 0; });
  // [一-龥぀-ゟ゠-ヿ]+ (std::wregex on the codepoints)
  auto cjk = [&](size_t i) {
    const uint32_t c = cp[i];
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
  };
  p = split(p, [&](size_t i, size_t e) { return run(i, e, cjk) - i; });
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

  std::vector<uint32_t> out(p.size());
  for (size_t k = 0; k < p.size(); k++) out[k] = uint32_t(p[k].second);
  return out;
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

BpeEncoder::BpeEncoder(const std::vector<std::string>& tokens, const std::vector<std::string>& merges) {
  std::unordered_map<std::string, int32_t> id;
  id.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) id.emplace(tokens[i], int32_t(i));
  for (int b = 0; b < 256; b++) {
    auto it = id.find(byte_char(b));  // (bytes no UTF-8 text holds, 0xC0, 0xC1, 0xF5 ..: maybe none)
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
