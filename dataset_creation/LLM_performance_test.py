import pandas as pd
from jinja2 import Template
from OpenRouter import OpenRouter

import numpy as np

class LLM_performance_test(object):
    def __init__(self):
        # Initialize any variables or settings
        pass




    def is_LFQA(self,router,prompt):
        response = router.get_response(prompt)
        print(prompt, response)
        if 'yes' in response.lower():
            #print(prompt, response)
            return True
        else:
            return False



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

    def update_response_llm(self,router):
        input_file = r'C:\Users\rafid\source\repos\Open_router_api\data\lfqa_dataset_post_veri_5_external_judgement_majority_vote_llm.xlsx'
        df = pd.read_excel(input_file)
        new_column_values = [None] * len(df['question_text'])

        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_short.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        prompt_file_path2 = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_batch.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path2, 'r', encoding='utf-8') as file:
            LFQA_filter_template2 = file.read()


        template = Template(LFQA_filter_template2)
        

        i = 0
        batch_size = 50

        while i < len(df):
            remaining = len(df) - i

            # Case 1: process 5 at once if 5 or more questions left
            if remaining >= batch_size+100:
                questions = df['question_text'][i:i+batch_size].tolist()
                #prompt = LFQA_filter_template2.format(*questions)
                prompt = template.render(questions=list(enumerate(questions, start=1)))
                response,content_logprobs = router.get_response_logprob(prompt)
                print(prompt, "\nresponse:\n", response)
                log_prob_tuple = self.log_prob_extractor(content_logprobs)
                answers = {}
                for line in response.strip().splitlines():
                    line = line.strip().lower()

                    for n in range(1, batch_size+1):
                        if f"{n}." in line and ("yes" in line or "no" in line):
                            answers[n] = 'yes' if 'yes' in line else 'no'

                for offset in range(batch_size):

                    if (offset <= len(log_prob_tuple)):
                        token, log_prob = log_prob_tuple[offset]
                        print(token,log_prob)

                        if (log_prob == 0):
                            new_column_values[i + offset] = answers.get(offset + 1, 'none')
                        else:
                            new_column_values[i + offset] = 'none'  # fill missing values
                            

                i += batch_size

            # Case 2: process one-by-one
            else:
                prompt = LFQA_filter_template.format(df['question_text'][i])
                response,content_logprobs = router.get_response_logprob(prompt)
                print(prompt,"single response:\n",response)
                print(self.log_prob_extractor(content_logprobs))
                token, log_prob = self.log_prob_extractor(content_logprobs)[0]
                print(token, log_prob)
   
                if ('yes' in response.lower()) and log_prob==0:
                    new_column_values[i] = 'yes'
                elif('no' in response.lower())and log_prob==0:
                    new_column_values[i] = 'no'
                else:
                    new_column_values[i] = 'none'
                i += 1
        df['llama-single-data-log'] = new_column_values
        df.to_excel(input_file, index=False)


    def calculate_match_percentages(self):
        file_path = r"F:\PhD\Long form research question\Preprocessing data\Post Data Verification\Follow 100 sample\evaluator responses\lfqa_dataset_post_veri_5_external_judgement_majority_vote.xlsx"
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        annotators = [col for col in df.columns if col not in ["question_text", "majority_vote"]]
        results = {}
    
        for annotator in annotators:
            matches = (df[annotator] == df["majority_vote"]).sum()
            total = len(df)
            print(f"{annotator} matches: {matches}, total: {total}")
            results[annotator] = round((matches / total) * 100, 2)
    
       
        percentages = results

        # Print results
        for annotator, pct in percentages.items():
            print(f"{annotator}: {pct}%")

        avg_percentage = round(sum(results.values()) / len(results), 2)
        print("Average: ", avg_percentage)





    def gwet_ac1(self):

        file_path = r"F:\PhD\Long form research question\Preprocessing data\Post Data Verification\Follow 100 sample\evaluator responses\lfqa_dataset_post_veri_5_external_judgement_majority_vote.xlsx"
        df = pd.read_excel(file_path, sheet_name="Sheet1")

        annotator_cols = [c for c in df.columns if c not in ["question_text", "majority_vote"]]
        #result = gwet_ac1(df[annotator_cols])


        df = pd.DataFrame(df[annotator_cols])
        n_items, n_raters = df.shape
        n_total = n_items * n_raters

        # Overall category proportions across all raters/items
        all_labels = pd.Series(df.values.ravel())
        p = all_labels.value_counts(normalize=True)
        m_bar = n_raters  # since each item has all raters in your case

        # Correct AC1 expected agreement
        Pe = (1.0 / (m_bar - 1.0)) * (1.0 - (p ** 2).sum())

        # Observed agreement Po
        pairwise_agreements = []
        for _, row in df.iterrows():
            counts = row.value_counts()
            num = (counts * (counts - 1)).sum()
            den = n_raters * (n_raters - 1)
            pairwise_agreements.append(num / den)

        Po = float(np.mean(pairwise_agreements))

        # AC1
        AC1 = (Po - Pe) / (1 - Pe) if Pe != 1 else np.nan

        print('Po:', Po, 'Pe:', Pe, 'AC1:', AC1, 'n_items:', n_items, 'n_ratings_total:', n_total)
