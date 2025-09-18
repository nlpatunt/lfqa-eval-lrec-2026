
from jinja2 import Template
from config.OpenRouter import OpenRouter

import pandas as pd

class Example_score:

    def __init__(self):
        # no setup here
        pass


    def is_example_response(self,response):
        if 'yes' in response.lower():
            return 1
        else:
            return 0


    def llm_response(self, router):
        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\zero_shot_example_detection.txt'

        # Read the prompt template
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()

        template = Template(LFQA_filter_template)

        file_path = r"F:\PhD\Long form research question\Example detector\eli5_qna_with_responses_gpt4o.xlsx"
        df = pd.read_excel(file_path)

        # Create a new column for responses
        df["gemini-2.5"] = None

        # Iterate using index
        for i in range(len(df)):
            question = df.loc[i, "question"]
            answer = df.loc[i, "answer"]  

            print(f"Row {i}:")

            # Format prompt
            prompt = LFQA_filter_template.format(question,answer)
            print("Prompt:", prompt)    

            response = router.get_response(prompt)
            print("Raw LLM Response:", response)    

            # Store binary classification
            df.loc[i, "gemini-2.5"] = self.is_example_response(response)
            print("LLM Response:", df.loc[i, "gemini-2.5"])

        # Save updated file
        output_path = r"F:\PhD\Long form research question\Example detector\eli5_qna_with_responses_gpt4o_gemini2.5.xlsx"
        df.to_excel(output_path, index=False)
        print(f"Saved updated file with LLM responses to {output_path}")

    def add_majority_vote(self ):
        in_path=r"F:\PhD\Long form research question\Example detector\eli5_qna_with_responses_gpt4o_gemini2.5.xlsx"
        out_path=r"F:\PhD\Long form research question\Example detector\eli5_qna_with_majority.xlsx"

        df = pd.read_excel(in_path)

        # get the last three columns (model outputs)
        model_cols = df.columns[-3:]

        def majority_vote(row):
            votes = row[model_cols].tolist()
            # pick the label that appears at least twice
            for val in set(votes):
                if votes.count(val) >= 2:
                    return val
            return "conflict"  # if all three disagree (shouldn’t happen with binary labels)

        df["majority_vote"] = df.apply(majority_vote, axis=1)

        if out_path:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                df.to_excel(w, index=False)


    def count_and_balance(self):

        in_path = r"F:\PhD\Long form research question\Example detector\eli5_qna_with_majority.xlsx"
       
        out_path= r"F:\PhD\Long form research question\Example detector\eli5_qna_with_majority_100.xlsx"
        col_name = r"majority_vote"
        sample_size = 50
        # Read Excel
        df = pd.read_excel(in_path)
        
        # Preview
        print("=== Preview of Data ===")
        print(df.head(), "\n")
        
        # Count distribution
        counts = df[col_name].value_counts().to_dict()
        print("=== Majority Vote Counts ===")
        for label, count in counts.items():
            print(f"{label}: {count}")
        
        # Balance: sample 50 from each class (0 and 1)
        df_1 = df[df[col_name] == 1].sample(n=sample_size, random_state=42)
        df_0 = df[df[col_name] == 0].sample(n=sample_size, random_state=42)
        balanced_df = pd.concat([df_1, df_0]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save new file
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            balanced_df.to_excel(writer, index=False, sheet_name="Balanced")
        
        print(f"\nBalanced dataset saved to {out_path}")
        return counts, balanced_df
        
