import json
class SHP_Dataset_Format:


    def shp_final_json_format(self):
        
        input_path = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge_lfqa.json" 
        output_path = r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge_lfqa_formatted.json" 

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
                "domain": "stack_stackoverflow",
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


        input_path =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge_unique.json" 

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


        input_path_all =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge.json"
        input_path_unique_lfqa =r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge_unique_lfqa.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\stackexchange\stack_stackoverflow\merge_lfqa.json" 



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


