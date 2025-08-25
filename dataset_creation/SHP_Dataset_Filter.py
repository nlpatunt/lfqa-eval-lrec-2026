from config.OpenRouter import OpenRouter
import json
import time
import os
import pandas as pd
from jinja2 import Template
import traceback
class SHP_Dataset_Filter:
    
    def is_LFQA(self,router,prompt):
        response = router.get_response(prompt)
        if 'yes' in response.lower():
            return True
        else:
            return False

    def filter_data(self,router):
        input_file = r"F:\PhD\Long form research question\SHP-2\stackexchange\askcarguys\merge_unique.json"
        output_file = r"F:\PhD\Long form research question\SHP-2\stackexchange\askcarguys\merge_unique_lfqa.json"

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
            #if t["token"].strip().lower() in ["yes", "no"]
            if t["token"].lstrip("Ġ ").lower() in ["yes", "no"]
        ]

        # Display the tuples
        #for i, (token, logprob) in enumerate(yes_no_logprobs, start=1):
            #print(f"{i}. Token: {token}, LogProb: {logprob}")
        return yes_no_logprobs

    def filter_data_unique(self,router):
        input_file = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_unique.json"
        output_file = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_unique_lfqa.json"
        output_file_label_log = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_unique_lfqa_label_log.json"

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\few_shot_instructions.txt'
        
        #Read all json data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            #data = json.load(f)
        print("Total: ", len(data))


        # Read Prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        prompt_file_path2 = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\few_shot_instructions_batch.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path2, 'r', encoding='utf-8') as file:
            LFQA_filter_template_batch = file.read()


        
        template = Template(LFQA_filter_template_batch)

        #Load existing filtered data
        data_lfqa = []
        log_zero_counter = 0
        data_lfqa_label_log = []

        # Load data_lfqa if file exists
        if os.path.exists(output_file):
           with open(output_file, 'r', encoding='utf-8') as f:
                data_lfqa = json.load(f)

        # Load data_lfqa_label_log if file exists
        if os.path.exists(output_file_label_log):
            with open(output_file_label_log, 'r', encoding='utf-8') as f:
                data_lfqa_label_log = json.load(f)

        

        i=0
        batch_size = 10
        try:
            while(i <  len(data)):
                time.sleep(0.02)
                remaining = len(data) - i
                print(remaining, i)

                if remaining >= batch_size:
                    questions = [item['history'] for item in data[i:i+batch_size]]
                    prompt = template.render(questions=list(enumerate(questions, start=1)))
                    data_batch = data[i:i+10]
                    response,content_logprobs = router.get_response_logprob(prompt)
                    
                    log_prob_tuple = self.log_prob_extractor(content_logprobs)
                    answers = {}
                    for line in response.strip().splitlines():
                        line = line.strip().lower()
                    

                        for n in range(1, batch_size+1):
                            if f"{n}." in line and ("yes" in line or "no" in line):
                                
                                if 'yes' in line:
                                    answers[n]= 'yes'
                                else:
                                    answers[n]= 'no'

                    
                    for offset in range(batch_size):

                        if (offset <= len(log_prob_tuple)):
                            token, log_prob = log_prob_tuple[offset]
                            

                            if (log_prob == 0 and (answers.get(offset + 1) == 'yes')):
                                data_lfqa.append(data_batch[offset])
                                log_zero_counter +=1
                            elif(log_prob == 0 and (answers.get(offset + 1) == 'no')):
                                log_zero_counter += 1
                            #Saving label log score
                            item_temp = data_batch[offset].copy()  # Avoid modifying original dataset
                            item_temp["label"] = answers.get(offset + 1)
                            item_temp["logscore"] = log_prob
                            data_lfqa_label_log.append(item_temp)
                        #else:
 

                    i += batch_size

            # Case 2: process one-by-one
                else:
                    prompt = LFQA_filter_template.format(data[i]['history'])
                    response,content_logprobs = router.get_response_logprob(prompt)
                   
                    token, log_prob = self.log_prob_extractor(content_logprobs)[0]
   
                    if ('yes' in response.lower()) and log_prob == 0:
                        data_lfqa.append(data[i])
                    

                    #Saving label log score
                    item_temp = data[i].copy()  # Avoid modifying original dataset
                    item_temp["label"] = token
                    item_temp["logscore"] = log_prob
                    data_lfqa_label_log.append(item_temp)
                    i += 1
            

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")

        except Exception as e:
            print("\nUnexpected error occurred:", e)
            traceback.print_exc()

        finally:
            print("New lfqa data inserted: ",len(data_lfqa))
            print("New lfqa data with label log inserted: ",len(data_lfqa_label_log))
            print("Log_zero",log_zero_counter)
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(data_lfqa, outfile, indent=2)


            with open(output_file_label_log, "w", encoding="utf-8") as outfile:
                json.dump(data_lfqa_label_log, outfile, indent=2)



    def filter_data_chatarena_lfqa_eval(self,router):
        input_file = r"F:\PhD\Long form research question\Preprocessing data\Attempt 3 with logscore\lfqa_pairwise_human_judgments_v1.json"
        output_file = r"F:\PhD\Long form research question\Preprocessing data\Attempt 3 with logscore\lfqa_pairwise_human_judgments_v1_logscore.json"

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\few_shot_instructions.txt'
        
        #Read all json data
        with open(input_file, 'r', encoding='utf-8') as f:
            #data = [json.loads(line) for line in f]
            data = json.load(f)
        print("Total: ", len(data))


        # Read Prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        prompt_file_path2 = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\few_shot_instructions_batch.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path2, 'r', encoding='utf-8') as file:
            LFQA_filter_template_batch = file.read()


        
        template = Template(LFQA_filter_template_batch)

        #Load existing filtered data
        data_lfqa = []
        log_zero_counter = 0
        data_lfqa_label_log = []

        # Load data_lfqa if file exists
        if os.path.exists(output_file):
           with open(output_file, 'r', encoding='utf-8') as f:
                data_lfqa = json.load(f)

        

        i=23550
        batch_size = 10
        try:
            while(i <  len(data)):

                if (data[i]['source'] in 'lfqa_eval'):
                    data_lfqa.append(data[i])
                    i+=1
                    print("lfqa_eval ",i)
                    continue
                    
                time.sleep(0.02)
                remaining = len(data) - i
                print(remaining, i)

                if remaining >= batch_size:
                    questions = [item['question_text'] for item in data[i:i+batch_size]]
                    prompt = template.render(questions=list(enumerate(questions, start=1)))
                    data_batch = data[i:i+10]
                    response,content_logprobs = router.get_response_logprob(prompt)
                    
                    log_prob_tuple = self.log_prob_extractor(content_logprobs)
                    answers = {}
                    for line in response.strip().splitlines():
                        line = line.strip().lower()
                    

                        for n in range(1, batch_size+1):
                            if f"{n}." in line and ("yes" in line or "no" in line):
                                
                                if 'yes' in line:
                                    answers[n]= 'yes'
                                else:
                                    answers[n]= 'no'

                    
                    for offset in range(batch_size):

                        if (offset <= len(log_prob_tuple)):
                            token, log_prob = log_prob_tuple[offset]
                            

                            if (log_prob == 0 and (answers.get(offset + 1) == 'yes')):
                                data_lfqa.append(data_batch[offset])
                                log_zero_counter +=1
                            elif(log_prob == 0 and (answers.get(offset + 1) == 'no')):
                                log_zero_counter += 1
                            #Saving label log score
                            item_temp = data_batch[offset].copy()  # Avoid modifying original dataset
                            item_temp["label"] = answers.get(offset + 1)
                            item_temp["logscore"] = log_prob
                            data_lfqa_label_log.append(item_temp)
                        #else:
 

                    i += batch_size

            # Case 2: process one-by-one
                else:
                    prompt = LFQA_filter_template.format(data[i]['question_text'])
                    response,content_logprobs = router.get_response_logprob(prompt)
                   
                    token, log_prob = self.log_prob_extractor(content_logprobs)[0]
   
                    if ('yes' in response.lower()) and log_prob == 0:
                        data_lfqa.append(data[i])
                    

                    #Saving label log score
                    item_temp = data[i].copy()  # Avoid modifying original dataset
                    item_temp["label"] = token
                    item_temp["logscore"] = log_prob
                    data_lfqa_label_log.append(item_temp)
                    i += 1
            

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")

        except Exception as e:
            print("\nUnexpected error occurred:", e)
            traceback.print_exc()

        finally:
            print("New lfqa data inserted: ",len(data_lfqa))
            print("New lfqa data with label log inserted: ",len(data_lfqa_label_log))
            print("Log_zero",log_zero_counter)
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(data_lfqa, outfile, indent=2)
