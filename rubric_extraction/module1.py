
import json
import os
from dotenv import load_dotenv 
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from datasets import get_dataset_config_names
class UniqueQuestionCounter:
    def count_unique_questions(self):
        file_path = r"F:\PhD\Long form research question\Final Dataset\large\lfqa_pairwise_human_judgments_v1_jsonl"
        questions = set()
        total = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    qtext = obj.get("question_text")
                    if qtext:
                        questions.add(qtext.strip())
                        total += 1

        print(f"[OK] Loaded {total:,} total records")
        print(f"[OK] Found {len(questions):,} unique questions")

    def remove_null_answers(self):
        # Step 1: Load all JSON objects
        input_path = r"F:\PhD\Long form research question\Final Dataset\large\lfqa_pairwise_human_judgments_v1_jsonl"
        output_path = r"F:\PhD\Long form research question\Final Dataset\large\LFQA-HP-1M"
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        print(f"[OK] Loaded {len(data):,} records")

        # Step 2: Filter out records with null or empty answers
        cleaned_data = [
            obj for obj in data
            if obj.get("answer_1") and obj.get("answer_2")
        ]

        removed = len(data) - len(cleaned_data)
        print(f"[OK] Removed {removed:,} records with null answers")

        # Step 3: Write cleaned list back to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for obj in cleaned_data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"[OK] Saved {len(cleaned_data):,} valid records to {output_path}")

    def load_dataset(self):
        """
        Loads a private Hugging Face dataset (JSONL files inside 'large' folder).
        """
        # --- 🔧 Configuration (edit if needed) ---
        data_files = "hf://datasets/nlpatunt/lfqa-pairwise-human-judgments/large/*.jsonl"
        env_path = Path(r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        load_dotenv(dotenv_path=env_path)
        token = os.getenv("HF_TOKEN")
        # ----------------------------------------

        try:
            # Authenticate
            print("🔐 Logging into Hugging Face Hub...")
            login(token)
            print("✅ Authentication successful.")

            # Load dataset
            #print(get_dataset_config_names("nlpatunt/lfqa-pairwise-human-judgments", token=token))
            #print(get_dataset_config_names("hf://datasets/nlpatunt/lfqa-pairwise-human-judgments"))  # shows config names
            data= load_dataset("nlpatunt/lfqa-pairwise-human-judgments", streaming=True,  token=token)


            print("✅ Dataset loaded successfully.")
            train_data = data["train"]

            for i, example in enumerate(train_data):
                print(example)  # each example is a Python dict (parsed JSON)
                if i >= 4:  # print only first 5
                    break
   
            return data

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return None

# --- Run directly ---
if __name__ == "__main__":
    uq = UniqueQuestionCounter()
    uq.load_dataset()
    
