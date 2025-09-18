import json
import os
class SHP_Dataset_Format:


    def shp_final_json_format(self):
        
        input_path = r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge_lfqa.json" 
        output_path = r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge_lfqa_formatted.json" 

        # Load original dataset
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        converted_data = []

        for idx, item in enumerate(data, start=1):
            new_item = {
                "question_id": item["post_id"].strip(),
                "question_text": item["history"].strip(),
                "answer_1": item["human_ref_A"].strip(),
                "answer_2": item["human_ref_B"].strip(),
                "human_judgment": "answer_1" if item["labels"] == 1 else "answer_2",
                "human_expert": False,
                "domain": "askvet",
                "language": None,
                "turn": None,
                "source": "shp-2-reddit"
            }
            converted_data.append(new_item)

        # Save the new dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)



    def filter_unique_post_ids(self):

        seen_post_ids = set()
        unique_entries = []


        input_path =r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge_unique.json" 

        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                post_id = data.get("post_id")
                if post_id not in seen_post_ids:
                    seen_post_ids.add(post_id)
                    unique_entries.append(data)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for entry in unique_entries:
                outfile.write(json.dumps(entry) + '\n')

        print(f"Saved {len(unique_entries)} unique post_id entries to {output_path}")

    def map_unique_lfqa_to_all_lfqa(self):


        input_path_all =r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge.json"
        input_path_unique_lfqa =r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge_unique_lfqa.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\reddit\askvet\merge_lfqa.json" 



        # Load the datasets
        with open(input_path_all, "r") as f:
            merge_dataset = [json.loads(line) for line in f]

        with open(input_path_unique_lfqa, "r") as f:
            merge_unique_lfqa = json.load(f)

        # Extract all unique post_ids from the unique dataset
        unique_post_ids = set(item['post_id'] for item in merge_unique_lfqa)

        # Filter merge_dataset based on unique post_ids
        filtered_dataset = [item for item in merge_dataset if item['post_id'] in unique_post_ids]

        # Save the filtered dataset
        with open(output_path, "w") as f:
            json.dump(filtered_dataset, f, indent=2)


    def merge_lfqa_json(self):


        root_dir = r"F:\PhD\Long form research question\SHP-2 - Merging\reddit"
        output_file = r"F:\PhD\Long form research question\SHP-2 - Merging\reddit\lfqa_pairwise_human_judgments_v1"
        merged_data = []
        i=0
        # Loop through each folder inside root_dir
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
        
            # Only process directories
            if os.path.isdir(folder_path):
                json_path = os.path.join(folder_path, "merge_lfqa_formatted.json")
                print(f"Processing {json_path}")
            
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            merged_data.extend(data)
                            i+=1
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")

        # Save merged JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f"Merged {len(merged_data)} objects into {output_file}","\n", i)



    def find_chatarena_lfqa_eval(self):
        
        # Everything hard-coded inside the function
        input_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1"                # Input JSON file
        output_file = r"F:\PhD\Long form research question\Final Dataset\chatarena_lfqa_eval"   # Output JSON file
        cutoff_qid = "q26032"                        # Cutoff question_id

        # Helper function scoped inside
        def qid_to_int(qid: str) -> int:
            qid = str(qid).strip().lower()
            if qid.startswith('q'):
                qid = qid[1:]
            try:
                return int(qid)
            except ValueError:
                return -1

        cutoff_num = qid_to_int(cutoff_qid)

        # Load → Filter → Save
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        filtered = [row for row in data if qid_to_int(row.get("question_id", "")) <= cutoff_num]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        print(f"Filtered {len(filtered)} items saved to {output_file}")

    def merge_json_files(self):
        # Hard-coded file names
        input_files = [
            r"F:\PhD\Long form research question\Final Dataset\chatarena_lfqa_eval",
            r"F:\PhD\Long form research question\SHP-2 - Merging\reddit\lfqa_pairwise_human_judgments_v1",
            r"F:\PhD\Long form research question\SHP-2 - Merging\stackexchange\lfqa_pairwise_human_judgments_v1"
        ]
        output_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1"

        merged_data = []

        # Read and combine
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {file} does not contain a list, skipping...")

        # Save merged result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f"Merged {len(merged_data)} items from {len(input_files)} files into {output_file}")



    def update_question_ids(self):
        input_file = r"F:\PhD\Long form research question\SHP-2 - Merging\reddit\lfqa_pairwise_human_judgments_v1"
        output_file = r"F:\PhD\Long form research question\SHP-2 - Merging\reddit\lfqa_pairwise_human_judgments_v11"
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        # Update question_id sequentially
        for idx, item in enumerate(data, start=1):
            item['question_id'] = f"q{idx}"
    
        # Save updated JSON back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
