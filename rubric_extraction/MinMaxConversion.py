
import json

class VeriScoreNormalizer:
    def __init__(self):
        pass

    def normalize_scores(self):
        """Normalize selected score fields to 1–5 scale and save updated JSONL file."""
        input_path = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test.jsonl"
        output_path = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_rescale.jsonl"

        updated_records = []

        # --- Normalization helper functions ---
        def scale_0_1_to_1_5(x):
            return 1.0 + (x * 4.0)

        def scale_1_3_to_1_5(x):
            return 1.0 + ((x - 1.0) / 2.0) * 4.0

        def scale_binary_to_1_5(x):
            return 5.0 if x >= 1 else 1.0

        # --- Field mappings by scale type ---
        fields_0_1 = [
            "veriscore_score_answer_1",
            "veriscore_score_answer_2",
            "conciseness_score_answer_1",
            "conciseness_score_answer_2"
        ]

        fields_1_3 = [
            "easy_to_understand_score_answer_1",
            "easy_to_understand_score_answer_2"
        ]

        fields_binary = [
            "example_score_answer_1",
            "example_score_answer_2"
        ]

        # --- Process file ---
        with open(input_path, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Apply scaling for 0–1 → 1–5
                for f in fields_0_1:
                    if f in data:
                        data[f] = scale_0_1_to_1_5(data[f])

                # Apply scaling for 1–3 → 1–5
                for f in fields_1_3:
                    if f in data:
                        data[f] = scale_1_3_to_1_5(data[f])

                # Apply scaling for binary (0 or 1) → 1 or 5
                for f in fields_binary:
                    if f in data:
                        data[f] = scale_binary_to_1_5(data[f])

                updated_records.append(data)

        # --- Save updated file ---
        with open(output_path, "w", encoding="utf-8") as outfile:
            for record in updated_records:
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")

        print(f"[OK] Normalized {len(updated_records)} records and saved to {output_path}")


if __name__ == "__main__":
    VeriScoreNormalizer().normalize_scores()
