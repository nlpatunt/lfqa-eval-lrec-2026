
from email import message
import sys, os,math,time


sys.path.append(r"C:\Users\rafid\Source\Repos\lfqa-eval")
from dataset_creation.Chatbot_Arena_Conversation_Dataset_Filter import Chatbot_Arena_Filter
from dataset_creation.LLM_performance_test import LLM_performance_test
from config.OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os
import json
import math
from scipy.stats import pearsonr
from numpy import dot
from numpy.linalg import norm
import numpy as np
import rubric_extraction.Completeness_score
import rubric_extraction.Easy_to_understand_score
import rubric_extraction.Specificity_score
import rubric_extraction.Example_score
import rubric_extraction.Well_structure_score

from dataset_creation.SHP_Dataset_Filter import SHP_Dataset_Filter
from dataset_creation.SHP_Dataset_format import SHP_Dataset_Format
import rubric_extraction.Specificity_score
import rubric_extraction.Grammar_score
import rubric_extraction.Completeness_score
import rubric_extraction.Well_structure_score
import rubric_extraction.Relevance_score
import rubric_extraction.Conciseness_score
import rubric_extraction.Example_score
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

class Rubric_based_evaluation:
    def __init__(self):
        # Initialize any variables or settings
        pass


    def load_data(self, file_path: str):
        """Load a JSONL file into a list of dicts."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    records.append(json.loads(line))
        print(f"[OK] Loaded {len(records)} records from {file_path}")
        return records

    def mutual_info(self):
        with open(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100_score_update", "r", encoding="utf-8") as f:
            data = json.load(f)  # expecting a JSON array of objects

        a1_orig, a1_rep = [], []
        a2_orig, a2_rep = [], []

        for obj in data:
            # Collect Answer 1
            if "veriscore_score_answer_1" in obj and "veriscore_replication_score_answer_1" in obj:
                v1 = obj["veriscore_score_answer_1"]
                r1 = obj["veriscore_replication_score_answer_1"]
                if v1 is not None and r1 is not None:
                    a1_orig.append(float(v1))
                    a1_rep.append(float(r1))

            # Collect Answer 2
            if "veriscore_score_answer_2" in obj and "veriscore_replication_score_answer_2" in obj:
                v2 = obj["veriscore_score_answer_2"]
                r2 = obj["veriscore_replication_score_answer_2"]
                if v2 is not None and r2 is not None:
                    a2_orig.append(float(v2))
                    a2_rep.append(float(r2))

        a1_orig = np.array(a1_orig, dtype=float)
        a1_rep  = np.array(a1_rep,  dtype=float)
        a2_orig = np.array(a2_orig, dtype=float)
        a2_rep  = np.array(a2_rep,  dtype=float)

        # mutual_info_regression expects X: 2D (features), y: 1D (target)
        # MI is symmetric in theory; estimator can vary slightly. We'll compute MI(X->y) with X = original, y = replication.
        mi_a1 = None
        mi_a2 = None

        if len(a1_orig) > 2:
            # Use a small k to be stable on modest sample sizes; set random_state for reproducibility.
            mi_a1 = mutual_info_regression(
                X=a1_orig.reshape(-1, 1),
                y=a1_rep,
                discrete_features=False,
                n_neighbors=3,
                random_state=42
            )[0]

        if len(a2_orig) > 2:
            mi_a2 = mutual_info_regression(
                X=a2_orig.reshape(-1, 1),
                y=a2_rep,
                discrete_features=False,
                n_neighbors=3,
                random_state=42
            )[0]

        print(f"Count A1 pairs: {len(a1_orig)} | Count A2 pairs: {len(a2_orig)}")
        print(f"Mutual Information (Answer 1: original → replication): {mi_a1}")
        print(f"Mutual Information (Answer 2: original → replication): {mi_a2}")



    def calculate(self):
        with open(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100_score_update", "r", encoding="utf-8") as f:
            data = json.load(f)   # assuming JSON array of objects

        scores1, rep_scores1 = [], []
        scores2, rep_scores2 = [], []

        for obj in data:
            if "veriscore_score_answer_1" in obj and "veriscore_replication_score_answer_1" in obj:
                scores1.append(obj["veriscore_score_answer_1"])
                rep_scores1.append(obj["veriscore_replication_score_answer_1"])

            if "veriscore_score_answer_2" in obj and "veriscore_replication_score_answer_2" in obj:
                scores2.append(obj["veriscore_score_answer_2"])
                rep_scores2.append(obj["veriscore_replication_score_answer_2"])

        # --- Answer 1 ---
        rmse1 = math.sqrt(sum((a - b) ** 2 for a, b in zip(scores1, rep_scores1)) / len(scores1))
        r1, _ = pearsonr(scores1, rep_scores1)
        cos1 = dot(scores1, rep_scores1) / (norm(scores1) * norm(rep_scores1))

        # --- Answer 2 ---
        rmse2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(scores2, rep_scores2)) / len(scores2))
        r2, _ = pearsonr(scores2, rep_scores2)
        cos2 = dot(scores2, rep_scores2) / (norm(scores2) * norm(rep_scores2))

        print(f"Answer 1 → RMSE: {rmse1:.4f}, Pearson r: {r1:.4f}, Cosine similarity: {cos1:.4f}")
        print(f"Answer 2 → RMSE: {rmse2:.4f}, Pearson r: {r2:.4f}, Cosine similarity: {cos2:.4f}")



    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="openai/gpt-4o",  # Replace with the model
            key=api_key
            
        )
        router_llama = OpenRouter(
            model_name="meta-llama/llama-4-scout",  # Replace with the model
            key=api_key
            
        )

        json_objects = self.load_data(r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_train.jsonl")
        print(len(json_objects))
        specificity_score = rubric_extraction.Specificity_score.Specificity_score()
        grammar_score = rubric_extraction.Grammar_score.Grammar_score()
        easy_to_understand_score = rubric_extraction.Easy_to_understand_score.Easy_to_understand_score()
        completeness_score = rubric_extraction.Completeness_score.Completeness_score()
        well_structure_score = rubric_extraction.Well_structure_score.Well_structure_score()
        relevance_score = rubric_extraction.Relevance_score.Relevance_score()
        conciseness_score = rubric_extraction.Conciseness_score.Conciseness_score()
        example_score = rubric_extraction.Example_score.Example_score()
        output_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_train_update.jsonl"


        i = 101
        while i < len(json_objects):
        #while i < 101:
            try:
                answer_1_score = (specificity_score.score("",json_objects[i]['question_text'],json_objects[i]['answer_1'])['score_1_5'])
                answer_2_score = (specificity_score.score("",json_objects[i]['question_text'],json_objects[i]['answer_2'])['score_1_5'])
                json_objects[i]['specificity_score_answer_1'] = answer_1_score
                json_objects[i]['specificity_score_answer_2'] = answer_2_score
                print(i,answer_1_score, answer_2_score)
                answer_1_score = grammar_score.score(json_objects[i]['answer_1'])
                answer_2_score = grammar_score.score(json_objects[i]['answer_2'])
                json_objects[i]['grammar_score_answer_1'] = answer_1_score
                json_objects[i]['grammar_score_answer_2'] = answer_2_score
                print(i,answer_1_score, answer_2_score)
                answer_1_score = easy_to_understand_score.calculate_score_fluency_textstat(router, json_objects[i]['answer_1'])
                answer_2_score = easy_to_understand_score.calculate_score_fluency_textstat(router, json_objects[i]['answer_2'])
                json_objects[i]['easy_to_understand_score_answer_1'] = answer_1_score
                json_objects[i]['easy_to_understand_score_answer_2'] = answer_2_score
                answer_1_score = (completeness_score.compute(router,json_objects[i]['question_text'],json_objects[i]['answer_1']))
                answer_2_score = (completeness_score.compute(router,json_objects[i]['question_text'],json_objects[i]['answer_2']))
                json_objects[i]['completeness_score_answer_1'] = answer_1_score
                json_objects[i]['completeness_score_answer_2'] = answer_2_score
                answer_1_score = well_structure_score.compute(router, json_objects[i]['question_text'],json_objects[i]['answer_1'])
                answer_2_score = well_structure_score.compute(router, json_objects[i]['question_text'],json_objects[i]['answer_2'])
                json_objects[i]['well_structure_score_answer_1'] = answer_1_score
                json_objects[i]['well_structure_score_answer_2'] = answer_2_score
                answer_1_score = relevance_score.compute(router, json_objects[i]['question_text'],json_objects[i]['answer_1'])
                answer_2_score = relevance_score.compute(router, json_objects[i]['question_text'],json_objects[i]['answer_2'])
                json_objects[i]['relevance_score_answer_1'] = answer_1_score
                json_objects[i]['relevance_score_answer_2'] = answer_2_score
                answer_1_score = conciseness_score.score(json_objects[i]['answer_1'])
                answer_2_score = conciseness_score.score(json_objects[i]['answer_2'])
                json_objects[i]['conciseness_score_answer_1'] = answer_1_score
                json_objects[i]['conciseness_score_answer_2'] = answer_2_score
                answer_1_score = (example_score.compute(router_llama,json_objects[i]['question_text'],json_objects[i]['answer_1']))
                answer_2_score = (example_score.compute(router_llama,json_objects[i]['question_text'],json_objects[i]['answer_2']))
                json_objects[i]['example_score_answer_1'] = answer_1_score
                json_objects[i]['example_score_answer_2'] = answer_2_score

                print(i)
                i += 1 

            except Exception as e:
                print(f" Error at index {i}: {e}")
                print("Retrying same index...")
                time.sleep(2)  # optional: wait before retry
                with open(output_file, "w", encoding="utf-8") as f:
                    for obj in json_objects:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                print(f"[OK] Updated JSONL saved to {output_file}")

                print(f"Updated JSON saved to {output_file}")



        # Save updated objects into a new file

        with open(output_file, "w", encoding="utf-8") as f:
            for obj in json_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] Updated JSONL saved to {output_file}")

        print(f"Updated JSON saved to {output_file}")

        

        

        

if __name__ == "__main__":
    app = Rubric_based_evaluation()
    app.run()

