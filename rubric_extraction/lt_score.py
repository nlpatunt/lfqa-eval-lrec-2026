import language_tool_python as lt

class LanguageToolScorer:
    def __init__(
        self,
        lang: str = "en-US", # Setting language to US English
        base_score: float = 100.0,#(out of 100).
        max_penalty_per_100_words: float = 25.0, #fairness adjustment — it normalizes scoring by text length, so scores reflect error density rather than just raw error count.
        category_weights: dict | None = None, #how severe each type of error is (e.g., typos penalize more than style).
        ignore_rule_ids: set | None = None,#et of rule IDs you don’t care about (can skip them).
    ):
        self.lang = lang
        self.base_score = base_score
        self.max_penalty_per_100_words = max_penalty_per_100_words
        # Weights categories of errors: how severe is each type of error
        self.category_weights = category_weights or {
            "TYPOS": 2.0,
            "GRAMMAR": 1.5,
            "PUNCTUATION": 1.0,
            "STYLE": 0.7,
            "CASING": 0.6,
            "REDUNDANCY": 0.6,
            "CONFUSED_WORDS": 1.6,
        }
        self.ignore_rule_ids = ignore_rule_ids or set()
        #  Actual tool call is happeningg here: Offline local engine (Java)
        self.tool = lt.LanguageTool(self.lang)

    def score(self, text: str) -> float:
        """Return only the score (0-100)."""
        #Runs LanguageTool on the text.
        #matches is a list of grammar/spelling/style issues.
        matches = [m for m in self.tool.check(text) if m.ruleId not in self.ignore_rule_ids]
        #Used to normalize penalties by text length.
        words = max(1, len(text.split()))
        raw_penalty = 0.0
        #Adds penalty based on the category_weights
        for m in matches:
            cat = str(getattr(m, "ruleIssueType", None) or getattr(m.category, "id", "OTHER")).upper()
            raw_penalty += self.category_weights.get(cat, 1.0)

        #Normalize for text length
        norm_factor = max(1.0, words / 100.0)
        #Example: for a 200-word text, penalty can be at most 2 × max_penalty_per_100_words.
        penalty_cap = self.max_penalty_per_100_words * norm_factor
        penalty = min(penalty_cap, raw_penalty)
        #Subtract penalties from base score (100).
        return round(max(0.0, self.base_score - penalty), 2)

# Example
#if __name__ == "__main__":
    #scorer = LanguageToolScorer()
    #txt = "NLP can refer to Natural Language Processing, an AI field where computers understand and generate human language, or Neuro-Linguistic Programming, a method for personal development and communication. "
    #rint("Score:", scorer.score(txt))
