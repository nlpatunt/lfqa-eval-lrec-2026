import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class PairwisePreferenceValidator:
    def __init__(self, dev_jsonl_path):
        self.dev_jsonl_path = dev_jsonl_path

        # ✅ Hard-coded learned weights (from training output)
        self.weights = {
            #"veriscore_score": 0.0524,
            #"veriscore_score": 0.0,
            #"specificity_score": 0.0420,
            #"specificity_score": 0.0,
            #"grammar_score": -0.0378,
            #"grammar_score": 0.0,
            #"easy_to_understand_score": 0.0919,
            #"easy_to_understand_score": 0.0,
            #"completeness_score": 0.4065,
            #"well_structure_score": 0.5910,
            #"relevance_score": -0.3082,
            #"relevance_score": 0.0,
            #"conciseness_score": 0.1747,
            #"example_score": 0.1208,
            
            #"specificity_score"             : 0.0488,
            #"grammar_score"                 : -0.0332,
            #"easy_to_understand_score"      : 0.0552,
            #"completeness_score"            : 0.3593,
            #"well_structure_score"          : 0.4576,
            #"relevance_score"               : -0.4609,
            #"conciseness_score"             : 0.1651,
            #"example_score"                 : 0.1210,
            #"factuality_geval_score"        : 0.4273

            "veriscore_score"               : 0.0524,
            "specificity_score"             : 0.0420,
            "grammar_score"                 : -0.0378,
            "easy_to_understand_score"      : 0.0919,
            "completeness_score"            : 0.4065,
            "well_structure_score"          : 0.5910,
            "relevance_score"               : -0.3082,
            "conciseness_score"             : 0.1747,
            "example_score"                 : 0.1208,
            
        }
        self.bias = 0.0275
        self.aspects = list(self.weights.keys())

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def load_dev_data(self):
        with open(self.dev_jsonl_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records

    def predict_on_dev(self, records):
        y_true, y_pred = [], []

        for r in records:
            truth = r.get("human_judgment")
            if truth not in ["answer_1", "answer_2"]:
                continue  # ❌ exclude ties

            x_diff = np.array([
                r.get(f"{a}_answer_1", 0) - r.get(f"{a}_answer_2", 0)
                for a in self.aspects
            ])

            w_vec = np.array([self.weights[a] for a in self.aspects])
            z = np.dot(w_vec, x_diff) + self.bias
            pA = self.sigmoid(z)

            # Predict preference
            pred = "answer_1" if pA >= 0.5 else "answer_2"

            y_true.append(truth)
            y_pred.append(pred)

        return y_true, y_pred

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["answer_1", "answer_2"], average=None
        )

        labels = ["answer_1", "answer_2"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print("\n📊 Validation Results (Ties Excluded)")
        print(f"Total examples: {len(y_true)}")
        print(f"Overall Accuracy: {acc:.4f}")
        print("\nPer-Class Metrics:")
        for i, lbl in enumerate(labels):
            print(f"  {lbl}: Precision={precision[i]:.4f} | Recall={recall[i]:.4f} | F1={f1[i]:.4f}")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(f"{labels}")
        print(cm)

    def run(self):
        records = self.load_dev_data()
        y_true, y_pred = self.predict_on_dev(records)
        self.evaluate(y_true, y_pred)


if __name__ == "__main__":
    # 🔹 Update with your actual dev JSONL path
    dev_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_rescale.jsonl"
    validator = PairwisePreferenceValidator(dev_file)
    validator.run()
