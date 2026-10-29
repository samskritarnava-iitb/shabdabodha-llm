"""
Deterministic Part-2 converter: turns a Part-1 "SB of the sentence" bracket
string (e.g. "(वर्तमान_प्रनि-((...)_प्रनि-...)_वि)") into the traditional
one-line gloss (e.g. "वर्तमानकालिकः ... गमनानुकूलव्यापारः").

This is a pure tree transformation, not a learned model — see the plan
discussion for why. The rules below are reverse-engineered from the single
worked hint example the user gave, plus general Sanskrit word-formation
rules (taddhita "-इक" suffix, "-अनुकूल-व्यापार" for the verbal head). They
are almost certainly incomplete for the full 200+ sentence dataset (passive
voice, participles, vocative, dual/plural persons, unseen karakas, etc.).
Treat KARAKA_VOCAB / PURUSHA_MAP / TENSE_MAP as the extension points and
validate/extend them against more of your labelled sentences before relying
on this for anything beyond a first-pass draft.
"""

import json
import sys

KARAKA_VOCAB = {
    "कर्तृ", "कर्म", "गौणकर्म", "मुख्यकर्म", "करण",
    "अपादान", "सम्प्रदान", "अधिकरण", "विशेषण",
}

PURUSHA_MAP = {
    "प्रपु-एव": "प्रथमपुरुषैकवचन",
    "मपु-एव": "मध्यमपुरुषैकवचन",
    "उपु-एव": "उत्तमपुरुषैकवचन",
}

TENSE_OVERRIDES = {
    "आज्ञा": "आज्ञार्थकः",
}

MATRA_CHARS = set("ािीुूृेैोौंःँ्")


def strip_prakr(word):
    return word[len("प्रकृ-"):] if word.startswith("प्रकृ-") else word


def add_ik_suffix(word):
    # attach the taddhita "-इक" suffix to a word ending in a bare
    # (inherent-'a') consonant, e.g. काल -> कालिक
    return word[:-1] + word[-1] + "ि" + "क"


def render_tense(label):
    if label in TENSE_OVERRIDES:
        return TENSE_OVERRIDES[label]
    base = label if "काल" in label else label + "काल"
    return add_ik_suffix(base) + "ः"


def sandhi_a_plus_a(word, suffix):
    # word ends in a bare consonant (inherent 'a'), suffix starts with "अ"
    assert suffix.startswith("अ")
    if word and word[-1] not in MATRA_CHARS:
        return word + "ा" + suffix[1:]
    return word + suffix


def render_head(word):
    return sandhi_a_plus_a(word, "अनुकूल") + "व्यापारः"


def split_top(inner):
    """Split 'LEFT_प्रनि-RIGHT_वि' (or the unhyphenated 'LEFT_प्रनि_RIGHT_वि'
    variant used for विशेषण/adjective links) at the paren-depth-0 separator."""
    depth = 0
    for sep in ("_प्रनि-", "_प्रनि_"):
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and inner[i:i + len(sep)] == sep:
                left = inner[:i]
                rest = inner[i + len(sep):]
                if rest.endswith("_वि"):
                    return left, rest[:-len("_वि")]
            i += 1
        depth = 0
    raise ValueError(f"no top-level separator found in {inner!r}")


def parse(node_str):
    node_str = node_str.strip()
    if not (node_str.startswith("(") and node_str.endswith(")")):
        return node_str  # bare token
    left, right = split_top(node_str[1:-1])
    return (parse(left), parse(right))


def is_self_identity_node(n):
    return (isinstance(n, tuple) and len(n) == 2
            and isinstance(n[0], str) and isinstance(n[1], str)
            and n[0].startswith("प्रकृ-") and n[1].startswith("त्यर्थ-"))


def is_purusha_karta_node(n):
    return (isinstance(n, tuple) and len(n) == 2
            and isinstance(n[0], str) and isinstance(n[1], str)
            and n[0] in PURUSHA_MAP and n[1] == "कर्तृ")


def try_resolve_leaf_group(node):
    left, right = node
    if is_self_identity_node(left) and is_purusha_karta_node(right):
        word = strip_prakr(left[0])
        purusha_phrase = PURUSHA_MAP[right[0]]
        return f"{word}-अभिन्न-{purusha_phrase}-कर्तृकः"
    if isinstance(left, str) and isinstance(right, str) and right in KARAKA_VOCAB:
        return f"{strip_prakr(left)}{right}कः"
    return None


def resolve_chain(node):
    if isinstance(node, str):
        return [render_head(node)]
    seg = try_resolve_leaf_group(node)
    if seg is not None:
        return [seg]
    left, right = node
    return resolve_chain(left) + resolve_chain(right)


def sb_sentence_to_gloss(sb_sentence):
    tree = parse(sb_sentence)
    tense_label, chain = tree
    segments = [render_tense(tense_label)] + resolve_chain(chain)
    return " ".join(segments)


if __name__ == "__main__":
    hint_example = (
        "(वर्तमान_प्रनि-((प्रकृ-वन_प्रनि-कर्म_वि)_प्रनि-(((प्रकृ-राम_प्रनि-त्यर्थ-राम_वि)"
        "_प्रनि-(प्रपु-एव_प्रनि-कर्तृ_वि)_वि)_प्रनि-गमन_वि)_वि)_वि)"
    )
    print("=== sanity check against the hint example ===")
    print(sb_sentence_to_gloss(hint_example))
    print("expected (structure): वर्तमानकालिकः <वन>कर्मकः राम-अभिन्न-प्रथमपुरुषैकवचन-कर्तृकः गमनानुकूलव्यापारः")

    print("\n=== बालः पादाभ्यां गृहं गच्छति। ===")
    sb = ("(वर्तमान_प्रनि-(((प्रकृ-बाल_प्रनि-त्यर्थ-बाल_वि)_प्रनि-(प्रपु-एव_प्रनि-कर्तृ_वि)_वि)"
          "_प्रनि-((पाद_प्रनि-करण_वि)_प्रनि-((गृह_प्रनि-कर्म_वि)_प्रनि-गमन_वि)_वि)_वि)_वि)")
    print(sb_sentence_to_gloss(sb))

    if len(sys.argv) > 1:
        print(f"\n=== spot-checking {sys.argv[1]} ===")
        with open(sys.argv[1], encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                ex = json.loads(line)
                sb_sentence = ex["sb_sentence"] if "sb_sentence" in ex else json.loads(ex["contents"][1]["parts"][0]["text"])["sb_sentence"]
                try:
                    gloss = sb_sentence_to_gloss(sb_sentence)
                except Exception as exc:
                    gloss = f"<parse failed: {exc}>"
                print(f"{i}: {gloss}")
