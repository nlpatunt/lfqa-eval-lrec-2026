import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PairwisePreferenceLogReg:
    def __init__(self, jsonl_path, include_ties=False):
        """
        include_ties=True  → keep tie samples with label=0.5 (ignored in training)
        include_ties=False → skip tie samples (default)
        """
        self.jsonl_path = jsonl_path
        self.include_ties = include_ties
        self.aspects = [
            "veriscore_score",
            "specificity_score",
            "grammar_score",
            "easy_to_understand_score",
            "completeness_score",
            "well_structure_score",
            "relevance_score",
            "conciseness_score",
            "example_score",
            #"factuality_geval_score"
        ]
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def load_data(self):
        """Load JSONL file"""
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records

    def prepare_features(self, records):
        """Compute aspect differences and map preferences"""
        X, y = [], []
        skipped_ties = 0

        for r in records:
            # Compute feature difference: A - B
            x_diff = []
            for aspect in self.aspects:
                a_key = f"{aspect}_answer_1"
                b_key = f"{aspect}_answer_2"
                x_diff.append(r.get(a_key, 0) - r.get(b_key, 0))

            pref = r.get("human_judgment")
            if pref == "answer_1":
                label = 1
            elif pref == "answer_2":
                label = 0
            elif pref == "tie":
                if self.include_ties:
                    label = 0.5
                else:
                    skipped_ties += 1
                    continue
            else:
                continue  # skip unknown label

            X.append(x_diff)
            y.append(label)

        if skipped_ties > 0:
            print(f"⚠️ Skipped {skipped_ties} tie cases (set include_ties=True to keep them).")

        return np.array(X), np.array(y)

    def train(self, X, y):
        """Train logistic regression"""
        # Exclude 0.5 labels since sklearn doesn't support fractional y
        mask = y != 0.5
        X, y = X[mask], y[mask]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        weights = dict(zip(self.aspects, self.model.coef_[0]))
        bias = float(self.model.intercept_[0])

        print("\n📈 Learned Weights per Aspect:")
        for k, v in weights.items():
            print(f"  {k:<30}: {v:.4f}")
        print(f"\nBias term: {bias:.4f}")
        return weights, bias

    def run(self):
        records = self.load_data()
        X, y = self.prepare_features(records)
        weights, bias = self.train(X, y)
        return weights, bias


if __name__ == "__main__":
    input_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_rescale.jsonl"
    model = PairwisePreferenceLogReg(input_file, include_ties=False)  # change to True to keep ties
    model.run()
