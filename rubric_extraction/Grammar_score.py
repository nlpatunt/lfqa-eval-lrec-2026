import language_tool_python as lt

class Grammar_score:
    def __init__(
        self,
        lang: str = "en-US",
        max_penalty_per_100_words: float = 5.0,  # max penalty per 100 words
        penalty_per_error: float = 1.0,          # penalty for each grammar error
        ignore_rule_ids: set | None = None,
    ):
        self.lang = lang
        self.max_penalty_per_100_words = max_penalty_per_100_words
        self.penalty_per_error = penalty_per_error
        self.ignore_rule_ids = ignore_rule_ids or set()
        self.tool = lt.LanguageTool(self.lang)

    def score(self, text: str) -> float:
        """Return grammar-only score on a 1.0–5.0 scale (decimals allowed)."""
        matches = [
            m for m in self.tool.check(text)
            if (m.ruleId not in self.ignore_rule_ids)
            and (
                getattr(m, "ruleIssueType", "").lower() == "grammar"
                or getattr(m.category, "id", "").upper() == "GRAMMAR"
            )
        ]
        words = max(1, len(text.split()))

        # raw penalty based on grammar error count
        raw_penalty = len(matches) * self.penalty_per_error

        # normalize by text length
        norm_factor = max(1.0, words / 100.0)
        penalty_cap = self.max_penalty_per_100_words * norm_factor
        penalty = min(penalty_cap, raw_penalty)

        # scale to 1–5
        # 5 = no errors, 1 = maximum penalty
        score = 5.0 - (penalty / penalty_cap) * 4.0

        # keep within range
        return round(max(1.0, min(5.0, score)), 2)


# Example usage
if __name__ == "__main__":
    scorer = Grammar_score()
    txt_good = "This sentence is written correctly and has no grammar mistakes."
    txt_bad = "This are bad sentence with many error that make no sense."

    print("Good text score:", scorer.score(txt_good))  # e.g. 4.8–5.0
    print("Bad text score:", scorer.score(txt_bad))    # e.g. 1.2–1.5
