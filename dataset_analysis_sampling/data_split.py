import json

from collections import Counter, defaultdict
from pathlib import Path
class DataSplit:
    
    def convert_json_to_jsonl(cls):
        input_path = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1"
        output_path = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_jsonl"
        # Load JSON
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If the JSON is a single object, wrap it into a list
        if isinstance(data, dict):
            data = [data]

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[OK] Loaded: {len(data)} records")
        print(f"[OK] Saved to: {output_path}")

    def analyze(cls):
        records = []
        file_path = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_sample_test"
        # load jsonl
        # Load JSONL
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Count per source
        source_counts = Counter(rec.get("source") for rec in records)

        # Count per (source, human_expert)
        source_expert_counts = Counter(
            (rec.get("source"), rec.get("human_expert")) for rec in records
        )

        # Count per domain
        domain_counts = Counter(rec.get("domain") for rec in records)

        # Unique domains per source
        domains_per_source = defaultdict(set)
        for rec in records:
            domains_per_source[rec.get("source")].add(rec.get("domain"))

        domains_per_source_count = {
            src: len(domains) for src, domains in domains_per_source.items()
        }

        # Print analysis
        print("=== Count per source ===")
        for k, v in source_counts.items():
            print(f"{k}: {v}")

        print("\n=== Count per (source, human_expert) ===")
        for k, v in source_expert_counts.items():
            print(f"{k}: {v}")

        print("\n=== Count per domain ===")
        for k, v in domain_counts.items():
            print(f"{k}: {v}")

        print("\n=== Unique domains per source ===")
        for k, v in domains_per_source_count.items():
            print(f"{k}: {v}")

        print(f"\n[OK] Total records: {len(records)}")



    
    def split(self):
        file_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval_shp_final_jsonl")  # fixed input path
        out_train = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_sample_train")
        out_dev   = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_sample_dev")
        out_test  = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_sample_test")
        records = []

        # load jsonl
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # separate last 602 for test append
        base_records = records[:4473]
        extra_test = records[4473:]

        # helper function: deterministic split by slicing
        def split_data(data):
            n = len(data)
            n_train = int(0.7 * n)
            n_dev = int(0.15 * n)
            n_test = n - n_train - n_dev
            return (
                data[:n_train],
                data[n_train:n_train+n_dev],
                data[n_train+n_dev:]
            )

        # storage
        train, dev, test = [], [], []

        # --- Group 1: human_expert=True, source="Chatbot Arena"
        group = [r for r in base_records if r.get("human_expert") and r.get("source") == "Chatbot Arena"]
        tr, dv, ts = split_data(group)
        train += tr; dev += dv; test += ts

        # --- Group 2: human_expert=True, source="lfqa_eval"
        group = [r for r in base_records if r.get("human_expert") and r.get("source") == "lfqa_eval"]
        tr, dv, ts = split_data(group)
        train += tr; dev += dv; test += ts

        # --- Group 3: human_expert=False, source="Chatbot Arena"
        group = [r for r in base_records if not r.get("human_expert") and r.get("source") == "Chatbot Arena"]
        tr, dv, ts = split_data(group)
        train += tr; dev += dv; test += ts

        # --- Group 4: human_expert=False, source="lfqa_eval"
        group = [r for r in base_records if not r.get("human_expert") and r.get("source") == "lfqa_eval"]
        tr, dv, ts = split_data(group)
        train += tr; dev += dv; test += ts

        # --- Group 5: shp-2-reddit + shp-2-stackexchange (per-domain split)
        group = [r for r in base_records if r.get("source") in {"shp-2-reddit", "shp-2-stackexchange"}]

        # group records by domain
        domain_map = {}
        for rec in group:
            dom = rec.get("domain")
            domain_map.setdefault(dom, []).append(rec)

        # split each domain independently
        for dom, recs in domain_map.items():
            tr, dv, ts = split_data(recs)
            train += tr
            dev += dv
            test += ts

        # append last 602 directly to test
        test += extra_test

        # write outputs
        def write_jsonl(path, data):
            with open(path, "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        write_jsonl(out_train, train)
        write_jsonl(out_dev, dev)
        write_jsonl(out_test, test)

        print(f"[OK] Train: {len(train)}")
        print(f"[OK] Dev: {len(dev)}")
        print(f"[OK] Test: {len(test)} (includes +{len(extra_test)} extra)")
        print("[DONE] train.jsonl, dev.jsonl, test.jsonl written.")

    def split_into_eight(self):
        # Base path inside function
        base_path = Path(
            r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\train intermediate\lfqa_pairwise_human_judgments_v1_sample_train"
        )

        # Load JSONL (line by line)
        with open(base_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        total = len(data)
        print(f"Total records: {total}")

        # Calculate chunk size for 8 splits
        chunk_size = total // 8
        remainder = total % 8

        start = 0
        for i in range(8):
            # Distribute remainder across first few chunks
            end = start + chunk_size + (1 if i < remainder else 0)
            chunk = data[start:end]

            # Save each chunk as JSONL
            output_path = base_path.parent / f"{base_path.stem}_part{i+1}.jsonl"
            with open(output_path, "w", encoding="utf-8") as out_f:
                for record in chunk:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Saved {len(chunk)} records to {output_path}")
            start = end

    def merge_back(self):
        # Base path inside function
        base_path = Path(
            r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\train intermediate\lfqa_pairwise_human_judgments_v1_sample_train"
        )

        # Find all part files (sorted to maintain order)
        part_files = sorted(base_path.parent.glob(f"{base_path.stem}_part*.jsonl"))
        merged_path = base_path.parent / f"{base_path.stem}_merged.jsonl"

        total_records = 0
        with open(merged_path, "w", encoding="utf-8") as out_f:
            for part_file in part_files:
                with open(part_file, "r", encoding="utf-8") as f:
                    lines = [line for line in f if line.strip()]
                    total_records += len(lines)
                    out_f.writelines(lines)
                    print(f"Loaded {len(lines)} records from {part_file}")

        print(f"Merged total {total_records} records into {merged_path}")

def main():
    dataSplit = DataSplit()
    dataSplit.merge_back()
    

if __name__ == "__main__":
    main()

