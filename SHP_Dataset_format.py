import json
import os
class SHP_Dataset_Format:


    def shp_final_json_format(self):
        
        input_path = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_lfqa.json" 
        output_path = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_lfqa_formatted.json" 

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
                "domain": "stack_workplace",
                "language": None,
                "turn": None,
                "source": "shp-2-stackexchange"
            }
            converted_data.append(new_item)

        # Save the new dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)



    def filter_unique_post_ids(self):

        seen_post_ids = set()
        unique_entries = []


        input_path =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_unique.json" 

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


        input_path_all =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge.json"
        input_path_unique_lfqa =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_unique_lfqa.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_workplace\merge_lfqa.json" 



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
