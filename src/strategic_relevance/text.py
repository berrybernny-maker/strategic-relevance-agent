import re


_re_alnum = re.compile(r"[A-Za-z0-9_]+")
_re_cjk = re.compile(r"[\u4e00-\u9fff]+")


def normalize_text(text):
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def tokenize(text):
    t = normalize_text(text)
    tokens = []
    tokens.extend(_re_alnum.findall(t))
    for seg in _re_cjk.findall(t):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(_cjk_bigrams(seg))
    if not tokens:
        tokens = [x for x in re.split(r"\W+", t) if x]
    return tokens


def _cjk_bigrams(seg):
    if len(seg) < 2:
        return [seg]
    return [seg[i : i + 2] for i in range(len(seg) - 1)]
