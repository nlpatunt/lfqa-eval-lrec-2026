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
        input_file = r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge_unique.json"
        output_file = r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge_unique_lfqa.json"

        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_batch10.txt'
        
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
        


    def log_prob_extractor(self,content_logprobs):
                # Extract yes/no tokens with logprobs as tuples
        yes_no_logprobs = [
            (t["token"].strip().lower(), t["logprob"])
            for t in content_logprobs
            if t["token"].strip().lower() in ["yes", "no"]
        ]

        # Display the tuples
        #for i, (token, logprob) in enumerate(yes_no_logprobs, start=1):
            #print(f"{i}. Token: {token}, LogProb: {logprob}")
        return yes_no_logprobs

    def filter_data_unique(self,router):
        input_file = r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge_unique.json"
        output_file = r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge_unique_lfqa.json"

        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_short.txt'
        
        #Read all json data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            #data = json.load(f)
        print("Total: ", len(data))


        # Read Prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        prompt_file_path2 = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_batch10.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path2, 'r', encoding='utf-8') as file:
            LFQA_filter_template_batch = file.read()

        #Load existing filtered data
        data_lfqa = []
        log_zero_counter = 0
    

        i=0
        while(i <  len(data)):
            remaining = len(data) - i
            print(remaining)

            if remaining >= 10:
                questions = [item['history'] for item in data[i:i+10]]
                data_batch = data[i:i+10]
                prompt = LFQA_filter_template_batch.format(*questions)
                response,content_logprobs = router.get_response_logprob(prompt)
                print(prompt, "\nresponse:\n", response)
                log_prob_tuple = self.log_prob_extractor(content_logprobs)
                answers = {}
                for line in response.strip().splitlines():
                    line = line.strip().lower()
                    

                    for n in range(1, 11):
                        if f"{n}." in line and ("yes" in line or "no" in line):
                            print(f"Line: {line}, n: {n}")
                            if 'yes' in line:
                                answers[n]= 'yes'
                            else:
                                answers[n]= 'no'

                print(answers)
                for offset in range(10):

                    if (offset <= len(log_prob_tuple)):
                        token, log_prob = log_prob_tuple[offset]
                        print(token,log_prob)

                        if (log_prob == 0 and (answers.get(offset + 1) == 'yes')):
                            data_lfqa.append(data_batch[offset])
                            log_zero_counter +=1
                        elif(log_prob == 0 and (answers.get(offset + 1) == 'no')):
                            log_zero_counter += 1
                        #else:
 

                i += 10

            # Case 2: process one-by-one
            else:
                prompt = LFQA_filter_template.format(data[i]['history'])
                response,content_logprobs = router.get_response_logprob(prompt)
                print(prompt,"response:\n",response)
                print(self.log_prob_extractor(content_logprobs))
                token, log_prob = self.log_prob_extractor(content_logprobs)[0]
                print(token, log_prob)
   
                if ('yes' in response.lower()):
                    data_lfqa.append(data[i])
                i += 1
            


        print("New lfqa data inserted: ",len(data_lfqa))
        print("Log_zero",log_zero_counter)
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(data_lfqa, outfile, indent=2)