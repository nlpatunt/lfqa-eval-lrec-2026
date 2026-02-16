
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class RegressionPreferenceModel:
    def __init__(self, train_jsonl_path, dev_jsonl_path):
        self.train_jsonl_path = train_jsonl_path
        self.dev_jsonl_path = dev_jsonl_path
        self.aspects = [
            "veriscore_score",
            "specificity_score",
            "grammar_score",
            "easy_to_understand_score",
            "completeness_score",
            "well_structure_score",
            "relevance_score",
            "conciseness_score",
            "example_score"
        ]
        self.model = LinearRegression()

    def load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def prepare_regression_data(self, records):
        """Expand each (A,B) pair into two rows: (answer, features, target)"""
        X, y = [], []

        for r in records:
            pref = r.get("human_judgment")

            # Target scores
            if pref == "answer_1":
                target_a, target_b = 1.0, 0.0
            elif pref == "answer_2":
                target_a, target_b = 0.0, 1.0
            elif pref == "tie":
                target_a = target_b = 0.5
            else:
                continue

            # Feature vectors for each answer
            features_a = [r.get(f"{a}_answer_1", 0) for a in self.aspects]
            features_b = [r.get(f"{a}_answer_2", 0) for a in self.aspects]

            X.append(features_a)
            y.append(target_a)

            X.append(features_b)
            y.append(target_b)

        return np.array(X), np.array(y)

    def train(self):
        print("📘 Loading training data...")
        records = self.load_data(self.train_jsonl_path)
        X, y = self.prepare_regression_data(records)

        print(f"Training on {len(y)} samples with {len(self.aspects)} features each...")
        self.model.fit(X, y)
        print("✅ Model training complete.")

        # Show learned feature weights
        coef_dict = dict(zip(self.aspects, self.model.coef_))
        print("\n🔍 Learned Weights (Linear Regression):")
        for k, v in coef_dict.items():
            print(f"  {k:<30}: {v:.4f}")
        print(f"Bias term: {self.model.intercept_:.4f}")

    def evaluate(self):
        print("\n📘 Loading dev data...")
        records = self.load_data(self.dev_jsonl_path)

        y_true, y_pred = [], []
        for r in records:
            if r.get("human_judgment") not in ["answer_1", "answer_2"]:
                continue  # skip ties for evaluation

            # Predict quality scores for both answers
            fA = np.array([r.get(f"{a}_answer_1", 0) for a in self.aspects]).reshape(1, -1)
            fB = np.array([r.get(f"{a}_answer_2", 0) for a in self.aspects]).reshape(1, -1)

            scoreA = self.model.predict(fA)[0]
            scoreB = self.model.predict(fB)[0]

            pred = "answer_1" if scoreA > scoreB else "answer_2"
            y_pred.append(pred)
            y_true.append(r["human_judgment"])

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["answer_1", "answer_2"], average=None
        )
        labels = ["answer_1", "answer_2"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print("\n📊 Regression-Based Validation Results (Ties Excluded)")
        print(f"Total examples: {len(y_true)}")
        print(f"Overall Accuracy: {acc:.4f}")
        print("\nPer-Class Metrics:")
        for i, lbl in enumerate(labels):
            print(f"  {lbl}: Precision={precision[i]:.4f} | Recall={recall[i]:.4f} | F1={f1[i]:.4f}")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(f"{labels}")
        print(cm)

    def run(self):
        self.train()
        self.evaluate()


if __name__ == "__main__":
    train_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_train_update.jsonl"
    dev_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_update.jsonl"

    model = RegressionPreferenceModel(train_file, dev_file)
    model.run()
