import json
class SHP_Dataset_Format:
    def format_data(self):
        jsons = []
        json_path = "F:\PhD\Long form research question\SHP-2\stackexchange\stack_academia\merge.json"

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                jsons.append(item)
        print(len(jsons))



    def filter_unique_post_ids(self):

        seen_post_ids = set()
        unique_entries = []


        input_path =r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge.json"
        output_path= r"F:\PhD\Long form research question\SHP-2\reddit\askcarguys\merge_unique.json" 

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





