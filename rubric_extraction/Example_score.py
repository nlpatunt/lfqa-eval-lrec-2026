import re
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import pipeline, AutoTokenizer


class Example_score:

    EXEMPLIFIER_REGEX = re.compile(
        r"\b(for example|for instance|e\.g\.|such as|like|to illustrate|including)\b",
        flags=re.IGNORECASE
    )
    NUMBER_REGEX = re.compile(r"\d")

    # lightweight "specificity" signals (no marker words)
    PERCENT_REGEX = re.compile(r"\d+\s*%")
    YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")
    RANGE_REGEX = re.compile(r"\b(from|between)\b.*\b(to|and)\b", flags=re.IGNORECASE)
    MONEY_REGEX = re.compile(r"\$|USD|EUR|£|¥")
    MONTH_REGEX = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
        flags=re.IGNORECASE
    )
    PAREN_REGEX = re.compile(r"[()]")

    def __init__(
        self,
        nli_model_id: str = "roberta-large-mnli",
        device: Optional[int] = None,
        use_sentence_level_probe: bool = True,
        decision_threshold: float = 0.8,
        decision_strategy: str = "sentence_max",  # "sentence_max" or "global_max"
        max_top_sentences: int = 3,
        chunk_word_limit: int = 300,
        max_token_length: int = 512,              # <-- tokenizer + calls use this
        # heuristics (OFF by default for marker-free)
        use_marker_heuristic: bool = False,
        use_number_heuristic: bool = False,
        marker_bonus: float = 0.25,
        number_bonus: float = 0.15,
        max_total_bonus: float = 0.35,
        # tie-breaker when model is flat ~0.5
        tie_breaker: str = "specificity",         # "specificity" or "none"
        specificity_threshold: float = 0.30,
        # output shaping
        force_zero_when_negative: bool = True,
        binarize_probs: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.use_sentence_level_probe = use_sentence_level_probe
        self.decision_threshold = float(decision_threshold)
        self.decision_strategy = str(decision_strategy).lower()
        self.max_top_sentences = int(max_top_sentences)
        self.chunk_word_limit = int(chunk_word_limit)
        self.max_token_length = int(max_token_length)

        # heuristic settings
        self.use_marker_heuristic = bool(use_marker_heuristic)
        self.use_number_heuristic = bool(use_number_heuristic)
        self.marker_bonus = float(marker_bonus)
        self.number_bonus = float(number_bonus)
        self.max_total_bonus = float(max_total_bonus)

        # tie-breaker settings
        self.tie_breaker = str(tie_breaker).lower()
        self.specificity_threshold = float(specificity_threshold)

        # output shaping
        self.force_zero_when_negative = bool(force_zero_when_negative)
        self.binarize_probs = bool(binarize_probs)

        # device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        # tokenizer with explicit max length -> removes truncation warning
        self.tokenizer = AutoTokenizer.from_pretrained(
            nli_model_id,
            model_max_length=self.max_token_length,
            truncation=True,
            padding_side="right"
        )

        # zero-shot pipeline with that tokenizer
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model=nli_model_id,
            tokenizer=self.tokenizer,
            device=device
        )

        # labels / templates
        # GLOBAL (keep for reference / optional strategy)
        self.global_label_pos = "contains a concrete example"
        self.global_labels = [self.global_label_pos]  # one-vs-null
        self.global_hypothesis_template = "The answer is {label}."

        # SENTENCE (primary signal): one-vs-null label
        self.sent_label_pos = "a concrete example"
        self.sentence_labels = [self.sent_label_pos]
        self.sentence_hypothesis_template = "This sentence is {label}."

    # -----------------------------
    # Public API
    # -----------------------------
    def predict(self, question: str, answer: str) -> Dict[str, Any]:
        if not isinstance(question, str):
            question = "" if question is None else str(question)
        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)

        chunks = self._chunk_by_words(answer, self.chunk_word_limit)

        # global (one-vs-null) probs for reference / optional decision
        agg_p_contains = 0.0
        for chunk in chunks:
            p = self._global_prob_on_chunk(question, chunk)
            if p > agg_p_contains:
                agg_p_contains = p
        agg_p_no = 1.0 - agg_p_contains

        # heuristics (usually 0.0 with heuristics off)
        bonus = self._heuristic_bonus(answer)

        # sentence-level probe (one-vs-null)
        top_sentences: List[Tuple[str, float]] = []
        top_prob = 0.0
        if self.use_sentence_level_probe:
            all_sentences = self._probe_sentences(answer)
            top_sentences = all_sentences[:self.max_top_sentences]
            if all_sentences:
                top_prob = float(all_sentences[0][1])

        # compute adjusted prob by strategy
        if self.decision_strategy == "sentence_max":
            adjusted = top_prob + bonus if top_prob >= 0.5 else max(0.0, top_prob - bonus * 0.5)
        elif self.decision_strategy == "global_max":
            adjusted = self._apply_bonus(agg_p_contains, 1.0 - agg_p_contains, bonus)
        else:
            adjusted = top_prob

        contains = adjusted >= self.decision_threshold

        # tie-breaker when flat
        if (not contains) and self.tie_breaker == "specificity":
            best_spec = self._best_sentence_specificity(answer)
            if best_spec >= self.specificity_threshold:
                contains = True
                adjusted = max(adjusted, min(1.0, 0.55 + best_spec * 0.3))

        final_label = "CONTAINS_EXAMPLE" if contains else "NO_EXAMPLE"

        # final score
        if not contains and self.force_zero_when_negative:
            final_score = 0.0
        else:
            final_score = adjusted
            if final_score < 0.0:
                final_score = 0.0
            if final_score > 1.0:
                final_score = 1.0

        # binarize probs to mirror decision (clean downstream behavior)
        if self.binarize_probs:
            out_p_contains, out_p_no = (1.0, 0.0) if contains else (0.0, 1.0)
        else:
            out_p_contains, out_p_no = agg_p_contains, 1.0 - agg_p_contains

        reasons = self._make_rationale(answer, top_sentences if contains else [], contains)

        return {
            "label": final_label,
            "score": round(float(final_score), 4),
            "probs": {
                "contains_example": round(float(out_p_contains), 4),
                "no_example": round(float(out_p_no), 4),
            },
            "heuristic_bonus": round(float(bonus), 4),
            "top_example_sentences": top_sentences if contains else [],
            "rationale": reasons,
            "meta": {
                "num_chunks": len(chunks),
                "chunk_size_words": self.chunk_word_limit,
                "decision_strategy": self.decision_strategy,
                "top_sentence_prob": round(float(top_prob), 4)
            }
        }

    # -----------------------------
    # Chunking & splitting
    # -----------------------------
    def _chunk_by_words(self, text: str, words_per_chunk: int) -> List[str]:
        tokens = text.split()
        if len(tokens) <= words_per_chunk:
            return [text.strip()] if text.strip() else [""]
        chunks: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            j = i + words_per_chunk
            if j > n:
                j = n
            chunk = " ".join(tokens[i:j]).strip()
            if chunk:
                chunks.append(chunk)
            i = j
        if not chunks:
            chunks = [""]
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        sentences: List[str] = []
        for s in parts:
            s = s.strip()
            if s and not s.isspace():
                sentences.append(s)
        return sentences

    # -----------------------------
    # Heuristics & rationale
    # -----------------------------
    def _heuristic_bonus(self, answer: str) -> float:
        bonus = 0.0
        if self.use_marker_heuristic and answer and self.EXEMPLIFIER_REGEX.search(answer):
            bonus += self.marker_bonus
        if self.use_number_heuristic and answer and self.NUMBER_REGEX.search(answer):
            bonus += self.number_bonus
        if bonus > self.max_total_bonus:
            bonus = self.max_total_bonus
        return bonus

    def _make_rationale(self, answer: str, top_sentences: List[Tuple[str, float]], contains: bool) -> str:
        messages: List[str] = []
        if self.use_marker_heuristic and self.EXEMPLIFIER_REGEX.search(answer or ""):
            messages.append("Found exemplification cue (e.g., 'for example', 'e.g.', 'such as').")
        if self.use_number_heuristic and self.NUMBER_REGEX.search(answer or ""):
            messages.append("Contains numeric/detail cues.")
        if contains and top_sentences:
            messages.append("Highlighted top sentence(s) likely to be examples.")
        if not messages:
            messages.append("Sentence-level NLI decision.")
        return " ".join(messages)

    # -----------------------------
    # Zero-shot passes (one-vs-null + explicit truncation/padding)
    # -----------------------------
    def _global_prob_on_chunk(self, question: str, answer_chunk: str) -> float:
        text = f"Question: {question}\nAnswer: {answer_chunk}"
        try:
            result = self.zero_shot(
                sequences=text,
                candidate_labels=self.global_labels,              # ["contains a concrete example"]
                hypothesis_template=self.global_hypothesis_template,  # "The answer is {label}."
                multi_label=True,                                 # one label -> score in [0,1]
                truncation=True,
                max_length=self.max_token_length,
                padding=True
            )
            probs = self._parse_zero_shot(result)
            p = float(probs.get(self.global_label_pos, 0.0))
        except Exception:
            if self.verbose:
                print("Error during global classification on a chunk.")
            p = 0.0
        return p

    def _probe_sentences(self, answer: str) -> List[Tuple[str, float]]:
        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        results: List[Tuple[str, float]] = []
        for sentence in sentences:
            seq = f"Sentence: {sentence}"
            try:
                r = self.zero_shot(
                    sequences=seq,
                    candidate_labels=self.sentence_labels,              # ["a concrete example"]
                    hypothesis_template=self.sentence_hypothesis_template,  # "This sentence is {label}."
                    multi_label=True,                                   # one label -> score in [0,1]
                    truncation=True,
                    max_length=self.max_token_length,
                    padding=True
                )
                probs = self._parse_zero_shot(r)
                p_is_example = float(probs.get(self.sent_label_pos, 0.0))
            except Exception:
                p_is_example = 0.0
            results.append((sentence, p_is_example))

        self._sort_sentences_by_score_desc(results)
        return results

    # -----------------------------
    # Specificity tie-breaker (no marker words)
    # -----------------------------
    def _best_sentence_specificity(self, text: str) -> float:
        sentences = self._split_sentences(text)
        best = 0.0
        for s in sentences:
            sc = self._specificity_score(s)
            if sc > best:
                best = sc
        return best

    def _specificity_score(self, s: str) -> float:
        score = 0.0
        if self.NUMBER_REGEX.search(s):
            score += 0.20
        if self.PERCENT_REGEX.search(s):
            score += 0.10
        if self.YEAR_REGEX.search(s):
            score += 0.10
        if self.MONTH_REGEX.search(s):
            score += 0.10
        if self.MONEY_REGEX.search(s):
            score += 0.10
        if self.RANGE_REGEX.search(s):
            score += 0.10
        if self.PAREN_REGEX.search(s):
            score += 0.05
        if score > 1.0:
            score = 1.0
        return score

    # -----------------------------
    # Parsing & Sorting
    # -----------------------------
    def _parse_zero_shot(self, result: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected zero-shot result type: {type(result)}. Got: {result}")
        labels = result.get("labels")
        scores = result.get("scores")
        if labels is None or scores is None or len(labels) != len(scores) or len(labels) == 0:
            raise ValueError(f"Zero-shot output missing/mismatched 'labels'/'scores'. Raw: {result}")
        parsed: Dict[str, float] = {}
        for i in range(len(labels)):
            lbl = str(labels[i])
            try:
                sc = float(scores[i])
            except Exception:
                sc = 0.0
            parsed[lbl] = sc
        return parsed

    def _sort_sentences_by_score_desc(self, items: List[Tuple[str, float]]) -> None:
        # selection sort (explicit; no lambdas)
        n = len(items)
        for i in range(n):
            best_idx = i
            best_score = items[i][1]
            j = i + 1
            while j < n:
                current_score = items[j][1]
                if current_score > best_score:
                    best_idx = j
                    best_score = current_score
                j += 1
            if best_idx != i:
                tmp = items[i]
                items[i] = items[best_idx]
                items[best_idx] = tmp

    # -----------------------------
    # Bonus (used only in global strategy or tiny nudge)
    # -----------------------------
    def _apply_bonus(self, p_contains: float, p_no: float, bonus: float) -> float:
        adjusted = p_contains + bonus if p_contains >= p_no else p_contains - (bonus * 0.5)
        if adjusted < 0.0:
            adjusted = 0.0
        if adjusted > 1.0:
            adjusted = 1.0
        return adjusted

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":


    detector = Example_score(
        nli_model_id="roberta-large-mnli",   # try this backbone
        decision_strategy="sentence_max",
        decision_threshold=0.65,             # was 0.8; lower to catch marker-free examples
        use_sentence_level_probe=True,
        use_marker_heuristic=False,          # keep marker-free
        use_number_heuristic=True,           # allow numeric concreteness (still marker-free)
        tie_breaker="specificity",
        specificity_threshold=0.15,          # was 0.30; lower so “19th century”, “16-day” counts
        binarize_probs=True,
        force_zero_when_negative=True
    )
    q = "Explain the impact of industrialization on society."
    essay = (
        "Industrialization led to rapid urban growth and changes in labor. "
        "For example, in 19th century Britain, textile factories expanded employment but also created poor working conditions. "
        "Child labor became widespread, which illustrated both economic opportunity and exploitation. "
        "In addition, industrialization boosted transport, as railways connected cities and markets across the country. "
        "Over time, legislation improved safety, but the early period showed stark social costs."
    )

    print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    print("---- ESSAY ----")
    print(detector.predict(q, essay))


    q = "Why did engagement drop last month?"

    a_with_marker = (
        "Daily active users fell 12% (from 1.2M to 1.05M). "
        "For example, a 16-day iOS push outage from Aug 2-18 reduced notification CTR by 34%, "
        "which drove fewer sessions and lower interactions."
    )

    a_without_marker = (
        "Engagement declined after a 16-day iOS push outage in August that reduced notification CTR by 34%, "
        "leading to fewer sessions and lower feed interactions overall."
    )

    a_no_example = (
        "Engagement dropped due to multiple operational issues and seasonal effects. "
        "The underlying factors were complex and varied across cohorts."
    )

    print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    print("---- WITH MARKER ----")
    print(detector.predict(q, a_with_marker))

    print("\n---- WITHOUT MARKER (still concrete) ----")
    print(detector.predict(q, a_without_marker))

    print("\n---- NO EXAMPLE ----")
    print(detector.predict(q, a_no_example))
