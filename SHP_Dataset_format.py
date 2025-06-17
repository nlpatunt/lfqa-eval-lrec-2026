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




