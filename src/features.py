"""Numeric features for KIO routing (final model).

Two cue scores built from GENERIC customer-service vocabulary (word stems), written once before
training and not tuned on the test set:
  x1 = support_score  : words about an existing problem (error, broken, leak, late, warranty...)
  x2 = sales_score    : words about a new purchase (quote, price, buy, lease, demo, package...)
Negation: a cue that appears within 3 words after no/not/don't/nothing/never is counted as
the opposite intent's evidence removed (e.g. "no issues", "don't want to buy").
"""
import re

SUPPORT_STEMS = ["error", "alarm", "fault", "fail", "broke", "broken", "stop", "down", "leak", "wrong",
    "late", "delay", "arriv", "missing", "miss", "repair", "fix", "technician", "engineer", "warrant",
    "replace", "complain", "issue", "problem", "noise", "vibrat", "overheat", "burn", "scorch", "crack",
    "lock", "login", "access", "twice", "refund", "invoice", "billed", "charged", "correct", "help",
    "existing", "already", "since", "after", "keeps", "again", "can't", "cannot", "won't", "doesn't",
    "don't fit", "not working", "differ", "match"]
SALES_STEMS = ["quote", "pric", "cost", "buy", "purchas", "order", "lease", "leasing", "financ", "discount",
    "bulk", "demo", "catalog", "brochure", "upgrade", "package", "plan$", "plans$", "subscription", "proposal",
    "propose", "offer", "sell", "available", "lead time", "deliver", "new", "add", "expand", "another",
    "second", "third", "interested", "training", "licen", "install", "turnkey", "how much"]
NEG = {"no", "not", "don't", "dont", "nothing", "never", "isn't", "without"}

def _hits(word, stems):
    """A stem matches the start of a word; a stem ending in $ must match the whole word."""
    for s in stems:
        if " " in s:
            continue
        if s.endswith("$"):
            if word == s[:-1]:
                return True
        elif word.startswith(s):
            return True
    return False

def featurize(text):
    t = text.lower()
    words = re.findall(r"[a-z0-9']+", t)
    sup = sal = 0.0
    neg = 0
    for w in words:
        if w in NEG:
            neg = 3
            continue
        s_hit, p_hit = _hits(w, SUPPORT_STEMS), _hits(w, SALES_STEMS)
        if neg > 0:          # negated cue: "no issues" / "don't want to buy"
            if s_hit: sup -= 1
            if p_hit: sal -= 1
        else:
            sup += s_hit
            sal += p_hit
        neg = max(0, neg - 1)
    for phrase in [s for s in SUPPORT_STEMS if " " in s]:
        sup += t.count(phrase)
    for phrase in [s for s in SALES_STEMS if " " in s]:
        sal += t.count(phrase)
    return [sup, sal]
