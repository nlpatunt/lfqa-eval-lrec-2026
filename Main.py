from email import message

from Chatbot_Arena_Conversation_Dataset_Filter import Chatbot_Arena_Filter
from LLM_performance_test import LLM_performance_test
from OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os

from SHP_Dataset_Filter import SHP_Dataset_Filter
from SHP_Dataset_format import SHP_Dataset_Format

class Main:
    def __init__(self):
        # Initialize any variables or settings
        pass

    def process_questions_with_router(self, router, prompt_template, questions):

        full_prompt = prompt_template + "\n"
        for idx, question in enumerate(questions):
             full_prompt += f"\nQuestion {idx}: {question}\nAnswer {idx}:\n"

        print(full_prompt)
        response = router.get_response(full_prompt)

        lines = response.splitlines()

        # Build the final result
        results = []
        for idx, question in enumerate(questions):
            answer = None
            for line in lines:
                if f"answer {idx}" in line.strip().lower():
                    answer = line.split(":", 1)[-1].strip()
                    break
            results.append({"question": question, "answer": answer})

        return results

    def run(self):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")

        router = OpenRouter(
            model_name="meta-llama/llama-4-scout",  # Replace with the model
            key=api_key
            
        )



        # Path to your JSONL file
        file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions.txt'

        # Read the JSONL file line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

  
  
        #arena_filter = Chatbot_Arena_Filter()
        #arena_filter.filter_data(router)
        #sHP_Dataset_Filter = SHP_Dataset_Filter()
        #sHP_Dataset_Filter.filter_data(router)
        lLM_performance_test = LLM_performance_test()
        lLM_performance_test.update_response_llm(router)
        #print(router.get_response(LFQA_filter_template.format("What is the meaning of life")))

        #sHP_Dataset_Format= SHP_Dataset_Format()
        #sHP_Dataset_Format.format_data()

if __name__ == "__main__":
    app = Main()
    app.run()

