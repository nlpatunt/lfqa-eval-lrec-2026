from email import message

from dataset_creation.Chatbot_Arena_Conversation_Dataset_Filter import Chatbot_Arena_Filter
from dataset_creation.LLM_performance_test import LLM_performance_test
from config.OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os


from rubric_extraction import Relevance_score
import rubric_extraction.Coherence_score

from dataset_creation.SHP_Dataset_Filter import SHP_Dataset_Filter
from dataset_creation.SHP_Dataset_format import SHP_Dataset_Format
import rubric_extraction.Relevance_score

class Main:
    def __init__(self):
        # Initialize any variables or settings
        pass

    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="openai/chatgpt-4o-latest",  # Replace with the model
            key=api_key
            
        )
        #arena_filter = Chatbot_Arena_Filter()
        #arena_filter.filter_data(router)

        #lLM_performance_test = LLM_performance_test()
        #lLM_performance_test.update_response_llm(router)
        #lLM_performance_test.calculate_match_percentages()
        #lLM_performance_test.gwet_ac1()

        #sHP_Dataset_Format= SHP_Dataset_Format()
        #sHP_Dataset_Format.filter_unique_post_ids() 

        #sHP_Dataset_Filter = SHP_Dataset_Filter()
        #sHP_Dataset_Filter.filter_data_unique(router)

        #sHP_Dataset_Format.map_unique_lfqa_to_all_lfqa()
        #sHP_Dataset_Format.shp_final_json_format()

        #ChatArena_LFQA_Eval
        #sHP_Dataset_Format.merge_lfqa_json()
        #sHP_Dataset_Format.update_question_ids()


        Relevance_score = rubric_extraction.Relevance_score.Relevance_score()

        Relevance_score.update_response_llm(router)

if __name__ == "__main__":
    app = Main()
    app.run()

