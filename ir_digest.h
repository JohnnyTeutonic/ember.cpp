#pragma once
// The digest of an IR document (paperkiln spec/IR_SPEC.md section 3.10, revision 1.1), written for ember
// from the specification text (no code taken from paperkiln):
//   1. the text is parsed as I-JSON (RFC 7493): no duplicate member names, every number a finite binary64,
//      every integer token within -2^53 .. 2^53, every string well-formed Unicode (no lone surrogates, valid
//      UTF-8);
//   2. every member whose name starts with "_" is removed, at any depth (kept when prose is true: a test of
//      the serialiser alone);
//   3. the rest is serialised by RFC 8785 (JCS): no whitespace, member names sorted by their UTF-16 code
//      units, strings as ECMAScript's JSON.stringify writes them (only '"', '\' and U+0000 .. U+001F escaped),
//      numbers as ECMAScript's Number.prototype.toString writes the binary64 value (shortest round-trip
//      digits, from std::to_chars);
//   4. the digest is SHA-256 (FIPS 180-4) of that text's UTF-8 bytes, 64 lowercase hex digits.
// A document that holds a negative zero has no digest (RFC 8785 writes -0 as 0). An integer token "-0" is the
// integer zero, not a negative zero (the suite's vector minus_zero_integer_form). The I-JSON conditions and
// the negative zero are checked on the whole parsed text, "_" members included (step 1 precedes step 2).
#include <string>

namespace ember_ir {

struct DigestResult {
  bool ok = false;         // false: the text has no digest (refused says why)
  std::string sha256;      // 64 lowercase hex digits
  std::string canonical;   // the RFC 8785 text (UTF-8)
  std::string refused;
};

DigestResult ir_digest(const std::string& text, bool prose = false);

// SHA-256 of bytes, 64 lowercase hex digits.
std::string sha256_hex_bytes(const std::string& bytes);

// ECMAScript Number::toString of a finite binary64 (exposed for the self-test).
std::string es_number_string(double v);

}  // namespace ember_ir
