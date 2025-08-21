from OpenRouter import OpenRouter
import json
import os
import pandas as pd
import traceback
import time

class Chatbot_Arena_Filter:
    
    def is_LFQA(self,router,prompt):
        try:
            response = router.get_response(prompt)
            if 'yes' in response.lower():
                return True
            else:
                return False
        except Exception as e:
            print("Error in API call:", e)
            traceback.print_exc()
            return None  # Indicates an error occurred

    def filter_data(self,router):
        input_file = r'C:\Users\rafid\source\repos\Open_router_api\data\lfqa_pairwise_human_judgments_v1.json'
        output_file = r'C:\Users\rafid\source\repos\Open_router_api\data\lfqa_pairwise_human_judgments_v1_3.json'
        existing_lfqa_file = r'C:\Users\rafid\source\repos\Open_router_api\data\lfqa_pairwise_human_judgments_v1_2.json'
        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions.txt'
        
        #Read all json data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Total: ", len(data))


        # Read Prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        #Load existing filtered data
        existing_lfqa = []
        if os.path.exists(existing_lfqa_file):
            with open(existing_lfqa_file, 'r', encoding='utf-8') as f:
                 existing_lfqa = json.load(f)
            print("mt_chatarba_lfqa File loaded successfully.")

        print("Existing filtered lfqa: ",len(existing_lfqa))
        start_index = 8428
        end_index = len(data)
        try:
            for i in range(start_index, end_index):
                item = data[i]
                if (item["source"] == "lfqa_eval"):
                    existing_lfqa.append(item)
                    continue
                prompt = LFQA_filter_template.format(item["question_text"])
                response = self.is_LFQA(router,prompt)
                if response is None:
                    print(f"Saving progress at index {i} due to error.")
                    break
                if response:
                    existing_lfqa.append(item)
                print(i)
                time.sleep(0.5) 

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")

        except Exception as e:
            print("\nUnexpected error occurred:", e)
            traceback.print_exc()

        finally:
            print("New lfqa data inserted: ",len(existing_lfqa))
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(existing_lfqa, outfile, indent=2)
        




