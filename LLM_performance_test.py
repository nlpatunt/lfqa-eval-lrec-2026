import pandas as pd
from OpenRouter import OpenRouter
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

    def update_response_llm(self,router):
        input_file = r'C:\Users\rafid\source\repos\Open_router_api\data\1 human 4 llms result.xlsx'
        df = pd.read_excel(input_file)
        new_column_values = [None] * len(df['question_text'])

        prompt_file_path = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_short.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        prompt_file_path2 = r'C:\Users\rafid\source\repos\Open_router_api\prompt\few_shot_instructions_batch10.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path2, 'r', encoding='utf-8') as file:
            LFQA_filter_template2 = file.read()
        

        i = 0

        while i < len(df):
            remaining = len(df) - i

            # Case 1: process 5 at once if 5 or more questions left
            if remaining >= 10:
                questions = df['question_text'][i:i+10].tolist()
                prompt = LFQA_filter_template2.format(*questions)
                response = router.get_response(prompt)
                print(prompt, "\nresponse:\n", response)

                answers = {}
                for line in response.strip().splitlines():
                    line = line.strip().lower()

                    for n in range(1, 11):
                        if f"{n}" in line and ("yes" in line or "no" in line):
                            answers[n] = 'yes' if 'yes' in line else 'no'

                for offset in range(10):
                    new_column_values[i + offset] = answers.get(offset + 1, 'none')

                i += 10

            # Case 2: process one-by-one
            else:
                prompt = LFQA_filter_template.format(df['question_text'][i])
                response = router.get_response(prompt)
                print(prompt,"response:\n",response)
   
                if ('yes' in response.lower()):
                    new_column_values[i] = 'yes'
                elif('no' in response.lower()):
                    new_column_values[i] = 'no'
                else:
                    new_column_values[i] = 'none'
                i += 1
        df['llama-4-final10x10_5'] = new_column_values
        df.to_excel(input_file, index=False)