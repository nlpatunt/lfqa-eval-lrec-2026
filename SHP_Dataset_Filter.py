from OpenRouter import OpenRouter
import json
import os
import pandas as pd
class SHP_Dataset_Filter:
    
    def is_LFQA(self,router,prompt):
        response = router.get_response(prompt)
        if 'yes' in response.lower():
            return True
        else:
            return False

    def filter_data(self,router):
        input_file = r'C:\Users\rafid\source\repos\Open_router_api\data\merge.json'
        output_file = r'C:\Users\rafid\source\repos\Open_router_api\data\merge_lfqa.json'

        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions.txt'
        
        #Read all json data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            #data = json.load(f)
        print("Total: ", len(data))


        # Read Prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        #Load existing filtered data
        data_lfqa = []
        post_flags = {}

        start_index = 0
        for i in range(start_index, len(data)):
            print(len(data) - i)
            item = data[i]
            if (item["post_id"] in post_flags):
                value = post_flags[item["post_id"]]
                if (value):
                     data_lfqa.append(item)
                continue


            prompt = LFQA_filter_template.format(item["history"])
            if self.is_LFQA(router,prompt):
                post_flags[item["post_id"]] = True
                data_lfqa.append(item)
            else:
                post_flags[item["post_id"]] = False
            


        print("New lfqa data inserted: ",len(data_lfqa))
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(data_lfqa, outfile, indent=2)
        




