#!/usr/bin/env python3
"""Build nfc_set_v1.jsonl: tokeniser test strings that NFC changes, for the Qwen NFC gate.

    python3 make_nfc_set.py [OUT]      # writes nfc_set_v1.jsonl beside this file (needs tokenizers)

Each row is {"i", "category", "text"}; NFC changes every text (Python's unicodedata and HF
tokenizers' NFC both; checked here). Code points are written {XXXX} (hex) in this file, so the
source is ASCII apart from ordinary composed words, which are decomposed with NFD before use.
Categories:
- decomposed: words in Latin-script languages, Vietnamese and Greek, written decomposed (NFD), and
  sentences wholly or half decomposed;
- reorder: combining marks out of canonical order, which NFC reorders (and then composes, or not);
- hangul: Korean written as conjoining jamo (L V, L V T, an LV syllable + T), jamo NFC cannot join
  (fillers, two L, two V, compatibility jamo) and Hangul tone marks;
- combining: combining sequences in other scripts (Devanagari, Bengali, Gurmukhi and Oriya nukta
  forms, which NFC decomposes; Tamil, Malayalam, Oriya and Kannada two-part vowels, which it
  composes; Arabic hamza and madda; Hebrew presentation forms; Tibetan; Cyrillic; several marks on
  one base) and canonical singletons (Angstrom, Ohm and Kelvin signs, the Greek question mark and
  ano teleia, the en and em quads, angle brackets, CJK compatibility ideographs, Greek tonos forms);
- compat: compatibility characters NFC must keep (ligatures, superscripts, full-width letters,
  circled and parenthesised forms, fractions, the trade mark sign, square units, Roman numerals, the
  long s, the micro sign, NBSP, the degree Celsius sign, Hangul compatibility jamo, digraphs) beside
  text NFC changes; checked to survive here, and in the gate by HF's ids;
- mixed: several scripts in one string, emoji and ZWJ sequences, digits, CR LF and tabs beside
  combining marks;
- specials: Qwen's added tokens beside text NFC changes, including "<", ">" and "=" before U+0338
  (NFC composes ">" + U+0338 to U+226F, so NFC must run between the added tokens, never across them).
"""
import json
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))


def U(s):
    """{XXXX} -> the code point."""
    return re.sub(r"[{]([0-9A-Fa-f]{4,6})[}]", lambda m: chr(int(m.group(1), 16)), s)


def D(s):
    return unicodedata.normalize("NFD", U(s))


WORDS = [
    "café", "naïve", "Zürich", "São Paulo", "Øresund og Århus", "Ångström", "crème brûlée", "Dvořák",
    "Łódź", "façade", "jalapeño", "piñata", "résumé", "über", "Müller", "Köln", "Straße mit Ärger",
    "Señor", "élève", "Noël", "coöperate", "Björk", "Ålesund", "Kraków", "Gdańsk", "Plzeň",
    "Brno-Líšeň", "Tiếng Việt", "Hà Nội", "Nguyễn", "phở bò", "Đà Nẵng", "cà phê sữa đá", "Ελληνικά",
    "ἄνθρωπος", "Ἀθῆναι", "ᾠδή", "φιλοσοφία", "Ῥόδος", "Iñtërnâtiônàlizætiøn", "Açaí", "São João",
    "Málaga", "Bogotá", "Medellín", "Perú", "Québec", "Montréal", "Genève", "Besançon", "Nîmes",
    "Orléans", "Ísland", "Reykjavík", "Þórshöfn", "Tórshavn", "Göteborg", "Malmö", "Århus",
    "Hämeenlinna", "Jyväskylä", "Tallinn ja Pärnu", "Šiauliai", "Rīga", "Ķekava", "Ljubljana in Žalec",
    "Čačak", "Niš", "Győr", "Pécs", "Brașov", "Timișoara", "İstanbul", "Eskişehir", "Muğla",
    "Çanakkale", "Ñuñoa", "Ðakovo i Županja", "Văn Miếu", "Huế", "Thừa Thiên", "Ỷ Lan",
    "{FB02}our and café",
]
SENTENCES = [
    "The café on the corner serves crème brûlée.", "Ich fahre morgen nach Zürich und Köln.",
    "¿Dónde está la estación de policía?", "Le garçon a mangé une pêche à Besançon.",
    "Tiếng Việt có sáu thanh điệu.", "Η Ελληνική γλώσσα έχει τόνους.", "Můžete mi pomoci, prosím?",
    "Zażółć gęślą jaźń.", "Příliš žluťoučký kůň úpěl ďábelské ódy.", "Árvíztűrő tükörfúrógép.",
    "Pchnąć w tę łódź jeża lub ośm skrzyń fig.", "Høj, kære venner, så går vi!",
    "Où est la bibliothèque, s'il vous plaît?", "Ação e reação são iguais.",
]
SINGLETONS = ["212B", "2126", "212A", "037E", "0387", "1FEF", "2000", "2001", "2329", "232A", "F900",
              "F91D", "FA10", "0340", "0341", "0343", "0374", "1F71", "1FBB", "2F800", "2F9D4"]
COMPAT = ["FB01", "FB02", "00B2", "00B9", "2460", "FF26", "00BD", "2122", "338F", "3231", "216B", "017F",
          "00B5", "00A0", "2103", "2075", "32C0", "FE5D", "1D2C", "24B6", "3392", "0132", "01C4", "2474",
          "3131", "FF76", "2153", "1E9B"]
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>"]
REORDER = ["a{0301}{0323}", "o{0302}{0323} phổ", "e{0301}{0327}", "u{031B}{0303}{0323}", "{1EA1}{0302}",
           "s{0307}{0323}", "{05E9}{05C2}{05BC}", "{0915}{094D}{093C}", "A{030A}{0316}",
           "{03B1}{0345}{0313}{0301}"]
HANGUL_WORDS = ["한국어", "안녕하세요", "감사합니다", "대한민국", "서울특별시", "김치찌개", "훈민정음", "읽다",
                "괜찮아요", "똑똑해"]
HANGUL_JAMO = ["{AC00}{11A8}", "{AC00}{11A8}{11A8}", "{1100}{1161}{11A8}{1100}{1161}", "{1100}{1161}{302E}",
               "{1112}{1175}{11AF}{302F}", "{115F}{1161} {1100}{1161}", "{1100}{1100}{1161}",
               "{1100}{1161}{1161}", "{3131}{314F} + {1100}{1161}"]
COMBINING = ["{0958}", "{0959}{093E}{0928}", "{095C}{0940}", "{09DC}", "{09DF}", "{0A33}", "{0B5C}",
             "{0B95}{0BC6}{0BBE}", "{0BC6}{0BD7}", "{0D46}{0D3E}", "{0D47}{0D3E}", "{0B47}{0B56}",
             "{0CBF}{0CD5}", "{0627}{0653}", "{0648}{0654}", "{064A}{0654}", "{0627}{0655}", "{FB2A}",
             "{FB4B}", "{FB1D}", "{0F73}", "{0F75}", "{0F43}", "e{0301}{0301}{0301}", "a{0308}{0304}",
             "u{0308}{0301}", "o{0303}{0301}", "{00C5}{0301}", "{0418}{0306}", "{0435}{0308}",
             "{0433}{0301}", "{0456}{0308}", "{04E9}{0308}"]
MIXED = ["Hello мир 世界 " + "{D}café" + " 한" + "{D}국", "{D}Ñandú" + " и " + "{D}ёлка" + " {1F642}",
         "1{0301} 2{0302} 3{0303} {D}ñ", " {0301}{0323} lead", "e{0301}{200D}{1F469}{200D}{1F4BB}",
         "{1F44D}{1F3FD} {D}bién", "{D}ελληνικά, русский, español", "tab{0009}{D}ü{000A}new",
         "日本語とKorean {D}한국어", "{05E9}{05B8}{05C1}{05DC}{05D5}{05B9}{05DD} {D}shalôm",
         "{0645}{0627}{0653}{0621} {D}mañana", "123{D}à456", "CAPS {D}ÉCOLE", "xxxxxxxxxxxxxxxxxxxxe{0301}",
         "{D}çà et là, où? ...", "{000D}{000A}{D}é{000D}{000A}", "{D}Ὀδυσσεύς (Odysseus)", "{2126}hm {D}ångström",
         "{F900} {D}crème", "{D}naïve{00A0}café"]
SPECIAL_EXTRA = ["<{0338}", ">{0338}", "={0338}", "a<{0338}b", "<|im_end|>={0338}", "<think>{D}é</think>",
                 "<|im_start|>user{000A}{D}Zürich<|im_end|>{000A}",
                 "<tool_call>{\"a\": \"{D}é\"}</tool_call>"]


def mixed(s):
    """'{D}' decomposes the rest of the space-free run after it (to the next space or code point)."""
    parts = re.split(r"([{]D[}])", s)
    out, decomp = "", False
    for p in parts:
        if p == "{D}":
            decomp = True
            continue
        if decomp:
            m = re.match(r"[^ {<]*", p)
            out += D(m.group(0)) + U(p[m.end():])
            decomp = False
        else:
            out += U(p)
    return out


def main():
    from tokenizers import normalizers
    hf = normalizers.NFC().normalize_str
    rng = random.Random(20261025)
    rows = []

    def add(cat, s):
        if unicodedata.normalize("NFC", s) == s or hf(s) == s:
            raise SystemExit(f"{cat}: NFC does not change {s!r}")
        rows.append({"i": len(rows), "category": cat, "text": s})

    for w in WORDS:
        add("decomposed", D(w))
    for s in SENTENCES:
        add("decomposed", D(s))
        add("decomposed", s[: len(s) // 2] + D(s[len(s) // 2:]))
    marks = ["0301", "0300", "0308", "0323", "0327", "0328", "031B", "0302", "0306", "0330", "0331", "0345",
             "05B4", "05BC", "0E38", "0E48", "093C", "094D"]
    for base in ["a", "e", "o", "u", "s", "q", "{03B1}", "{0438}", "x"]:
        for _ in range(4):
            s = U(base + "".join("{" + m + "}" for m in rng.sample(marks, 3)))
            if unicodedata.normalize("NFC", s) != s and hf(s) != s:
                add("reorder", s)
    for s in REORDER:
        add("reorder", U(s))
    for w in HANGUL_WORDS:
        add("hangul", D(w))
        add("hangul", "Korean: " + D(w) + "!")
    for _ in range(20):
        add("hangul", D("".join(chr(0xAC00 + rng.randrange(11172)) for _ in range(rng.randrange(1, 5)))))
    for s in HANGUL_JAMO:
        add("hangul", U(s))
    for s in COMBINING:
        add("combining", U(s))
    for c in SINGLETONS:
        add("combining", "x" + chr(int(c, 16)) + "y")
    add("combining", U("Temperature 300{212A} and 5{2126}, {212B}ngström"))
    for k, c in enumerate(COMPAT):
        ch = chr(int(c, 16))
        s = ch + " " + D(WORDS[k % len(WORDS)]) + " " + ch
        add("compat", s)
        assert unicodedata.normalize("NFC", s).count(ch) == 2 and hf(s).count(ch) == 2, c
    for s in MIXED:
        add("mixed", mixed(s))
    for sp in SPECIALS:
        add("specials", sp + chr(0x338))
        add("specials", U("e{0301}") + sp + chr(0x301))
        add("specials", D("café") + sp + D("crème"))
        add("specials", "x" + sp + chr(0x338) + sp + chr(0x301))
    for s in SPECIAL_EXTRA:
        add("specials", mixed(s))

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "nfc_set_v1.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(len(rows), cats)


if __name__ == "__main__":
    main()
