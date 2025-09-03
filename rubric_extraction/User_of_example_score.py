# example_detector_corenlp_collect.py
# Requires a running Stanford CoreNLP server with openie enabled.
import requests, re
from collections import defaultdict

class ExampleDetectorCoreNLP:
    def __init__(self, corenlp_url: str = "http://localhost:9000"):
        self.url = corenlp_url
        self.props = {
            "annotators": "tokenize,ssplit,pos,lemma,depparse,natlog,openie",
            "outputFormat": "json"
        }
        self._copula_tokens = ("be", "is", "are", "was", "were", "'s", "’s")
        self._include_prefix = ("include", "includes", "included", "including")

        # Regex fallbacks
        self._re_typeof = re.compile(
            r"\b(is|are|was|were)\s+(an?\s+)?(kind|type|sort|class)\s+of\s+\w",
            re.IGNORECASE
        )
        self._re_include_list = re.compile(
            r"\b(include|includes|including)\b[^.]*?\b\w+\b[^.]*?,[^.]*?\b(\w+|and\s+\w+)\b",
            re.IGNORECASE
        )
        self._re_all_tail = re.compile(r"(—|-|,)\s*all\s+\w+", re.IGNORECASE)

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _is_likely_category(np: str) -> bool:
        return len(re.findall(r"\b[A-Z][a-zA-Z]+\b", np)) <= 1

    @staticmethod
    def _split_coord_list(text: str):
        t = re.sub(r"\b(and|or)\b", ",", text, flags=re.IGNORECASE)
        return [i.strip(" ,.;:-—") for i in t.split(",") if i.strip()]

    def _sentence_text(self, sent) -> str:
        parts = []
        for tok in sent.get("tokens", []):
            parts.append(tok["originalText"])
            parts.append(tok.get("after", ""))
        return "".join(parts).strip()

    # ---------- text-level fallbacks ----------
    def _sentence_text_has_example(self, s: str) -> bool:
        if self._re_typeof.search(s):        # "is a type of ..."
            return True
        if self._re_include_list.search(s):  # "include x, y ..."
            return True
        if self._re_all_tail.search(s):      # "— all <category>"
            return True
        return False

    # ---------- tuple-level rules ----------
    def _tuple_signals_example(self, subj: str, rel: str, obj: str) -> bool:
        subj = self._norm(subj); rel = self._norm(rel.lower()); obj = self._norm(obj)

        # A) [list] + copula + [category]
        if any(tok in rel for tok in self._copula_tokens):
            items = self._split_coord_list(subj)
            if len(items) >= 2 and self._is_likely_category(obj):
                return True

        # B) "type/kind/sort/class of" anywhere
        composed = f"{subj} {rel} {obj}".lower()
        if re.search(r"\b(kind|type|sort|class)\b\s+of\s+\w", composed):
            return True

        # C) Category include(s) [list] (single tuple case)
        if rel.startswith(self._include_prefix):
            items = self._split_coord_list(obj)
            if len(items) >= 2 and self._is_likely_category(subj):
                return True

        # D) Object begins with "all <category>"
        if re.match(r"\ball\s+\w+", obj.lower()):
            items = self._split_coord_list(subj)
            if len(items) >= 2:
                return True

        return False

    # ---------- aggregation (handles CoreNLP splitting) ----------
    def _aggregated_signals_example(self, triples) -> bool:
        # 1) Multiple SUBJECTS share (rel,obj) → list → category
        by_rel_obj = defaultdict(set)
        # 2) Multiple OBJECTS share (subj,rel) with include(s) → category → list
        by_subj_rel = defaultdict(set)

        for t in triples:
            subj = self._norm(t["subject"])
            rel  = self._norm(t["relation"].lower())
            obj  = self._norm(t["object"])
            by_rel_obj[(rel, obj)].add(subj)
            by_subj_rel[(subj, rel)].add(obj)

        for (rel, obj), subjects in by_rel_obj.items():
            if len(subjects) >= 2 and any(tok in rel for tok in self._copula_tokens) and self._is_likely_category(obj):
                return True

        for (subj, rel), objects in by_subj_rel.items():
            if len(objects) >= 2 and rel.startswith(self._include_prefix) and self._is_likely_category(subj):
                return True

        return False

    # ---------- public APIs ----------
    def has_example(self, paragraph: str) -> int:
        """
        Return 1 if any sentence contains an example; else 0.
        """
        has, _ = self.collect_all_examples(paragraph)
        return 1 if has else 0

    def collect_all_examples(self, paragraph: str):
        """
        Return (has_any, matches), where matches is a list of:
          {"sentence": <text>, "evidence": "regex|tuple|aggregate"}
        Collects EVERY sentence that matches, useful for long paragraphs.
        """
        resp = requests.post(
            self.url,
            params={"properties": str(self.props)},
            data=paragraph.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        matches = []
        for sent in data.get("sentences", []):
            s_text = self._sentence_text(sent)
            triples = sent.get("openie", []) or []

            # 0) text regex first
            if self._sentence_text_has_example(s_text):
                matches.append({"sentence": s_text, "evidence": "regex"})
                continue  # no need to double-count

            # 1) any single tuple match
            fired = False
            for tr in triples:
                if self._tuple_signals_example(tr["subject"], tr["relation"], tr["object"]):
                    matches.append({"sentence": s_text, "evidence": "tuple"})
                    fired = True
                    break
            if fired:
                continue

            # 2) aggregated signals for this sentence
            if self._aggregated_signals_example(triples):
                matches.append({"sentence": s_text, "evidence": "aggregate"})

        return (len(matches) > 0, matches)


# --- quick demo ---
if __name__ == "__main__":
    det = ExampleDetectorCoreNLP()
    long_para = (
        "Learning algorithms can be very efficient in solving problems. "
        "They are used in many areas of computer science. "
        
    )
    has_any, matches = det.collect_all_examples(long_para)
    print("HAS ANY:", int(has_any))
    for m in matches:
        print(m["evidence"].upper(), "=>", m["sentence"])
