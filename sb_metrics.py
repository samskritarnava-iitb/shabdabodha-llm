import re
from collections import Counter

KARAKA = {"कर्तृ", "कर्म", "गौणकर्म", "मुख्यकर्म", "करण", "अपादान", "सम्प्रदान", "अधिकरण",
          "सम्बन्ध", "विशेषण", "भाव"}


def norm(s):
    return re.sub(r"\s+", "", s or "")


def balanced(s):
    d = 0
    for ch in s:
        d += ch == "("
        d -= ch == ")"
        if d < 0:
            return False
    return d == 0


def outer(s):
    m = re.match(r"\(?([^()_]+)", norm(s))
    return m.group(1) if m else ""


def head(s):
    m = re.findall(r"_प्रनि[-_]([^()_\s]+)_वि\)", norm(s))
    return m[-1] if m else ""


def labels(s):
    toks = re.findall(r"_प्रनि[-_]([^()_\s]+)_वि\)", norm(s))
    return Counter(t for t in toks if t in KARAKA or t.endswith("अधिकरण"))


def f1(a, b):
    tp = sum((a & b).values())
    if not a and not b:
        return 1.0
    p = tp / max(1, sum(b.values()))
    r = tp / max(1, sum(a.values()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score(preds, gts):
    n = len(gts)
    return {
        "balanced": sum(balanced(p) for p in preds) / n,
        "exact": sum(norm(p) == norm(g) for p, g in zip(preds, gts)) / n,
        "outer": sum(outer(p) == outer(g) for p, g in zip(preds, gts)) / n,
        "head": sum(head(p) == head(g) and head(g) != "" for p, g in zip(preds, gts)) / n,
        "f1": sum(f1(labels(g), labels(p)) for p, g in zip(preds, gts)) / n,
    }
