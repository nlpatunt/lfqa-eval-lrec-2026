import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class LengthBiasCheck:
    def test_length_bias(self):
        # --- File path (edit if needed) ---
        jsonl_path = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_0_temp.jsonl"
        # --- Models to analyze ---
        llm_fields = [
            "lLM_judgement_response_gpt4o_majority",
            "lLM_judgement_response_llama_majority",
            "lLM_judgement_response_gemini_majority"
        ]

        # --- Load JSONL data ---
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        print(f"[OK] Loaded {len(records)} records\n")

        # --- Iterate over LLM judgments ---
        for field in llm_fields:
            model_name = field.replace("lLM_judgement_response_", "").upper()

            total = 0
            bias_short = bias_long = 0
            agree = disagree = 0
            human_short_llm_long = 0
            human_long_llm_short = 0

            y_true, y_pred = [], []

            for obj in records:
                if field not in obj or obj[field] not in ["answer_1", "answer_2"]:
                    continue
                if "human_judgment" not in obj or obj["human_judgment"] not in ["answer_1", "answer_2"]:
                    continue

                total += 1
                human = obj["human_judgment"]
                llm_choice = obj[field]

                ans1_len = len(obj["answer_1"].split())
                ans2_len = len(obj["answer_2"].split())

                # Determine which is shorter/longer
                shorter = "answer_1" if ans1_len < ans2_len else "answer_2"
                longer = "answer_2" if ans1_len < ans2_len else "answer_1"

                # Count whether LLM prefers short or long
                if llm_choice == shorter:
                    bias_short += 1
                else:
                    bias_long += 1

                # Track normal agreement metrics
                y_true.append(human)
                y_pred.append(llm_choice)
                if llm_choice == human:
                    agree += 1
                else:
                    disagree += 1

                # --- NEW: Cross-bias cases ---
                # (1) human-correct answer was short, LLM picked long
                if human == shorter and llm_choice == longer:
                    human_short_llm_long += 1
                # (2) human-correct answer was long, LLM picked short
                if human == longer and llm_choice == shorter:
                    human_long_llm_short += 1

            # --- Metrics ---
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

            # --- Report ---
            print(f"===== LENGTH BIAS CHECK ({model_name}) =====")
            print(f"Total Samples                      : {total}")
            print(f"Short-Answer Preference             : {bias_short} ({bias_short/total:.3f})")
            print(f"Long-Answer Preference              : {bias_long} ({bias_long/total:.3f})")
            print(f"Human Agreement                     : {agree}/{total} ({agree/total:.3f})")
            print(f"Accuracy                            : {acc:.4f}")
            print(f"Precision                           : {prec:.4f}")
            print(f"Recall                              : {rec:.4f}")
            print(f"F1 Score                            : {f1:.4f}")
            print("--- Cross-Bias Cases ---")
            print(f"Human-Correct Short → LLM Picked Long : {human_short_llm_long}")
            print(f"Human-Correct Long  → LLM Picked Short: {human_long_llm_short}")
            print("-" * 60)

# --- Run ---
if __name__ == "__main__":
    checker = LengthBiasCheck()
    checker.test_length_bias()
