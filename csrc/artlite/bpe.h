// Byte-level BPE (GPT-2's) where gigatoken has no scanner for a vocabulary's pre-tokenizer (it always splits with one of
// its own): the pieces llama.cpp's regexes leave (src/llama-vocab.cpp, src/unicode.cpp's unicode_regex_split: each
// regex splits the previous one's pieces into its matches and the gaps between them), then the merges of each piece
// (llm_tokenizer_bpe_session: the lowest rank first, the leftmost of equal ranks).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace artlite {

// A codepoint's class as llama.cpp's regexes see it: a non-ASCII codepoint collapsed to its Unicode category (\p{N},
// \p{L}, \p{P}, \p{M}, \p{S}), whitespace before its category (Unicode's White_Space: \s), anything else OTHER; an
// ASCII one by the regexes' own ranges (the caller passes its class too: unicode_classes in models.tokenizer).
enum PreClass : uint8_t { PC_OTHER = 0, PC_N = 1, PC_L = 2, PC_P = 3, PC_M = 4, PC_S = 5, PC_WS = 6 };

// The pieces' lengths (in codepoints, in order, summing to n) of llama.cpp's "spark2_5" pre-tokenizer: \p{N}{1,3}, then
// [一-龥぀-ゟ゠-ヿ]+, then [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
// ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+, then \p{N}. cp: the codepoints, cls: their classes.
std::vector<uint32_t> pretokenize_spark2_5(const uint32_t* cp, const uint8_t* cls, size_t n);

// The same of llama.cpp's "flint" pre-tokenizer: \p{N}{1,3}, then [一-龥぀-ゟ゠-ヿ]+, then
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'s|'t|'re|'ve|'m|'ll|'d)?|...|
// ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ as llama.cpp runs it: std::regex over the categories, where the
// letter cases are one \p{L} (the contractions' letters either case).
std::vector<uint32_t> pretokenize_flint(const uint32_t* cp, const uint8_t* cls, size_t n);

// The same of llama.cpp's GPT-4o (o200k) pre-tokenizer ("gpt-4o", "inkling2"): one regex,
// [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'s|'t|'re|'ve|'m|'ll|'d)?|
// [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'s|...)?|\p{N}{1,3}|
// ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+ as llama.cpp runs it: std::regex over the categories, where only
// the ASCII letters have a case (a non-ASCII letter or mark is either), a mark is no \p{L} (the contractions' letters
// either case).
std::vector<uint32_t> pretokenize_gpt4o(const uint32_t* cp, const uint8_t* cls, size_t n);

// A vocabulary's merges as token ids.
class BpeEncoder {
 public:
  // tokens: the vocabulary's strings (GPT-2's byte-level form: a byte as one character, U+0120 for a space), merges:
  // "a b" in rank order. Throws if a merge's part or its result is not a token (a byte that is none: at its encode).
  // ignore_merges: a piece that is a token is that token, whatever the merges say (llama.cpp's ignore_merges).
  BpeEncoder(const std::vector<std::string>& tokens, const std::vector<std::string>& merges, bool ignore_merges = false);

  // The ids of a text's pieces: cp its codepoints, lens the pieces' lengths in codepoints (summing to n).
  std::vector<int32_t> encode(const uint32_t* cp, size_t n, const uint32_t* lens, size_t n_pieces) const;

 private:
  int32_t byte_id_[256];
  std::unordered_map<uint64_t, uint64_t> merges_;  // left id << 32 | right id -> rank << 32 | merged id
  std::unordered_map<std::string, int32_t> words_;  // (ignore_merges: the tokens by their byte-level strings)
  std::string byte_char_[256];                      // (a byte's character in those strings, UTF-8)
};

}  // namespace artlite
