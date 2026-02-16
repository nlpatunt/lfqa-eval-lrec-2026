from email import message

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



class Main:
    def __init__(self):
        # Initialize any variables or settings
        pass

    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="google/gemini-2.5-flash",  # Replace with the model
            key=api_key
            
        )
        #arena_filter = Chatbot_Arena_Filter()
        #arena_filter.filter_data(router)

        lLM_performance_test = LLM_performance_test()
        #lLM_performance_test.update_response_llm(router)
        lLM_performance_test.evaluate()
        #lLM_performance_test.gwet_ac1()

        #sHP_Dataset_Format= SHP_Dataset_Format()
        #sHP_Dataset_Format.filter_unique_post_ids() 

        #sHP_Dataset_Filter = SHP_Dataset_Filter()
        #sHP_Dataset_Filter.filter_data_unique(router)

        #sHP_Dataset_Format.map_unique_lfqa_to_all_lfqa()
        #sHP_Dataset_Format.shp_final_json_format()

        #ChatArena_LFQA_Eval
        #sHP_Dataset_Format.merge_lfqa_json()
        #sHP_Dataset_Format.merge_json_files()
        #sHP_Dataset_Format.update_question_ids()


        #Well_structure_score = rubric_extraction.Well_structure_score.Well_structure_score()

        #Well_structure_score.update_response_llm(router)

        #Easy_to_understand_score = rubric_extraction.Easy_to_understand_score.Easy_to_understand_score()

        #Easy_to_understand_score.calculate_score_fluency_textstat(router)


        #Completeness_score = rubric_extraction.Completeness_score.Completeness_score()

        #Completeness_score.test_performance_geval_yescieval(router)


        #example_score = rubric_extraction.Example_score.Example_score() 
        #example_score.count_and_balance()


        #specificity_score = rubric_extraction.Specificity_score.Specificity_score()
        #q = "Why did engagement drop last month?"
        #a = (
        #"My name is rafid. I studied at UNT.whatever happend is happend"
        #)
        #print(specificity_score.score(question=q, answer=a))


        

        

        

if __name__ == "__main__":
    app = Main()
    app.run()

