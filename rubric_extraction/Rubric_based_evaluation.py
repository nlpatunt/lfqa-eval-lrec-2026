
from email import message
import sys, os


sys.path.append(r"C:\Users\rafid\Source\Repos\lfqa-eval")
from dataset_creation.Chatbot_Arena_Conversation_Dataset_Filter import Chatbot_Arena_Filter
from dataset_creation.LLM_performance_test import LLM_performance_test
from config.OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os


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

class Rubric_based_evaluation:
    def __init__(self):
        # Initialize any variables or settings
        pass


    def load_data(self,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records



    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="openai/gpt-4.1-mini",  # Replace with the model
            key=api_key
            
        )

        json_objects = self.load_data(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100_score")
        print(len(json_objects))
        specificity_score = rubric_extraction.Specificity_score.Specificity_score()
        grammar_score = rubric_extraction.Grammar_score.Grammar_score()
        easy_to_understand_score = rubric_extraction.Easy_to_understand_score.Easy_to_understand_score()
        completeness_score = rubric_extraction.Completeness_score.Completeness_score()

        for i in range(len(json_objects)):
            #answer_1_score = (specificity_score.score("",json_objects[i]['question_text'],json_objects[i]['answer_1'])['score_1_5'])
            #answer_2_score = (specificity_score.score("",json_objects[i]['question_text'],json_objects[i]['answer_2'])['score_1_5'])
            #answer_1_score = grammar_score.score(json_objects[i]['answer_1'])
            #answer_2_score = grammar_score.score(json_objects[i]['answer_2'])
            #answer_1_score = easy_to_understand_score.calculate_score_fluency_textstat(router, json_objects[i]['answer_1'])
            #answer_2_score = easy_to_understand_score.calculate_score_fluency_textstat(router, json_objects[i]['answer_2'])
            answer_1_score = (completeness_score.compute(router,json_objects[i]['question_text'],json_objects[i]['answer_1']))
            answer_2_score = (completeness_score.compute(router,json_objects[i]['question_text'],json_objects[i]['answer_2']))
            

            json_objects[i]["completeness_score_answer_1"] = answer_1_score
            json_objects[i]["completeness_score_answer_2"] = answer_2_score
            print(i)




        # Save updated objects into a new file
        output_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100_score_update"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_objects, f, indent=2, ensure_ascii=False)

        print(f"Updated JSON saved to {output_file}")

        

        

        

if __name__ == "__main__":
    app = Rubric_based_evaluation()
    app.run()

