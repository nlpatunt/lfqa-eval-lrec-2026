import json
from pathlib import Path
import random
from collections import defaultdict
from collections import Counter
import json, re, html
class HumanFilter:
    
    def append_chatarena_samples(self):

        # --- file paths (edit if needed) ---
        full_path   = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
        expert_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_human_expert")
        output_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena")

        # --- load input files ---
        with full_path.open("r", encoding="utf-8") as f:
            all_records = json.load(f)
        with expert_path.open("r", encoding="utf-8") as f:
            expert_records = json.load(f)

        expert_ids = {rec.get("question_id") for rec in expert_records if rec.get("question_id")}

        # --- filter eligible candidates ---
        candidates = [
            rec for rec in all_records
            if rec.get("source") == "Chatbot Arena"
            and rec.get("human_expert") is False
            and rec.get("question_id") not in expert_ids
        ]

        if len(candidates) < 490:
            raise ValueError(f"Only {len(candidates)} eligible candidates, fewer than 705 requested.")

        # --- random sample ---
        random.seed(42)  # reproducible
        sampled = random.sample(candidates, 490)

        # --- append and save ---
        final = expert_records + sampled
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        # --- report ---
        print(f"[OK] Loaded full dataset: {len(all_records):,}")
        print(f"[OK] Human expert subset: {len(expert_records):,}")
        print(f"[OK] Eligible candidates (Chatbot Arena, human_expert=False): {len(candidates):,}")
        print(f"[OK] Sampled and appended: {len(sampled):,}")
        print(f"[OK] Final total: {len(final):,}")
        print(f"[OK] Saved to: {output_path}")


    def append_lfqa_Eval_samples(self):

        # --- file paths (edit if needed) ---
        full_path   = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
        expert_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena")
        output_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval")

        # --- load input files ---
        with full_path.open("r", encoding="utf-8") as f:
            all_records = json.load(f)
        with expert_path.open("r", encoding="utf-8") as f:
            expert_records = json.load(f)

        expert_ids = {rec.get("question_id") for rec in expert_records if rec.get("question_id")}

        # --- filter eligible candidates ---
        candidates = [
            rec for rec in all_records
            if rec.get("source") == "lfqa_eval"
            and rec.get("human_expert") is False
            and rec.get("question_id") not in expert_ids
        ]

        if len(candidates) < 490:
            raise ValueError(f"Only {len(candidates)} eligible candidates, fewer than 705 requested.")

        # --- random sample ---
        random.seed(42)  # reproducible
        sampled = random.sample(candidates, 490)

        # --- append and save ---
        final = expert_records + sampled
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        # --- report ---
        print(f"[OK] Loaded full dataset: {len(all_records):,}")
        print(f"[OK] Human expert subset: {len(expert_records):,}")
        print(f"[OK] Eligible candidates (Chatbot Arena, human_expert=False): {len(candidates):,}")
        print(f"[OK] Sampled and appended: {len(sampled):,}")
        print(f"[OK] Final total: {len(final):,}")
        print(f"[OK] Saved to: {output_path}")

    
    def append_domains_from_reddit_stackexchange(self):
        """
        Inputs:
          - FULL_DATASET_PATH: JSON array with all records
          - PREV_SAMPLE_PATH:  JSON array with previous sample (append target)
        Logic:
          - Consider only source in {"shp-2-reddit", "shp-2-stackexchange"}
          - Group by 'domain'
          - For each domain, randomly take up to 'per_domain_target' (default 40)
          - Exclude any record whose question_id is already in PREV_SAMPLE_PATH
        Output:
          - OUTPUT_PATH: previous sample + newly selected domain samples
        """
        FULL_DATASET_PATH   = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
        PREV_SAMPLE_PATH    = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval")
        OUTPUT_PATH         = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval_shp")
        per_domain_target = 25
        seed = 42
        sources_of_interest = {"shp-2-reddit", "shp-2-stackexchange"}
        random.seed(seed)


        with FULL_DATASET_PATH.open("r", encoding="utf-8") as f:
            all_records = json.load(f)
        with PREV_SAMPLE_PATH.open("r", encoding="utf-8") as f:
            prev_sample = json.load(f)

        prev_ids = {r.get("question_id") for r in prev_sample if r.get("question_id")}
        # Bucket eligible records by domain
        by_domain = defaultdict(list)
        for rec in all_records:
            if rec.get("source") not in sources_of_interest:
                continue
            qid = rec.get("question_id")
            if not qid or qid in prev_ids:
                continue
            dom = rec.get("domain")
            # Only consider proper domain strings (skip None just in case)
            if isinstance(dom, str) and dom:
                by_domain[dom].append(rec)

        # Sample per domain
        added = []
        domains_lt_target = []
        for dom, items in by_domain.items():
            k = min(per_domain_target, len(items))
            if k < per_domain_target:
                domains_lt_target.append((dom, len(items)))
            # If there are duplicates of question_id inside a domain list, guard against it
            # (rare, but safe)
            # Sample without replacement using random.sample
            picks = items if len(items) <= k else random.sample(items, k)
            # (if len(items) == k, items is fine; else sample)
            # But we also must ensure no duplicates across domains (shouldn't happen if domain is disjoint)
            added.extend(picks if len(items) <= k else random.sample(items, k))

        # Deduplicate across domains by question_id (paranoia)
        seen = set(prev_ids)
        dedup_added = []
        for rec in added:
            qid = rec.get("question_id")
            if qid and qid not in seen:
                dedup_added.append(rec)
                seen.add(qid)

        final = prev_sample + dedup_added

        # Save
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        # Reporting
        total_domains = len(by_domain)
        total_added = len(dedup_added)
        print(f"[OK] Domains considered from reddit/stackexchange: {total_domains}")
        print(f"[OK] Added {total_added} records (up to {per_domain_target} per domain)")
        if domains_lt_target:
            short = ", ".join(f"{d}({n})" for d, n in sorted(domains_lt_target))
            print(f"[INFO] Domains with < {per_domain_target}: {short}")
        print(f"[OK] New total after append: {len(final)} → {OUTPUT_PATH}")

    
    def append_repeats_groups_to_target(self):
        """
        Inputs (set inside the function):
          - full_dataset_path: JSON ARRAY with all records
          - prev_sample_path : JSON ARRAY with previously filtered/appended data
        Output:
          - output_path      : prev_sample + ~2,485 records formed by choosing whole qid-groups
                               from sources {"shp-2-reddit","shp-2-stackexchange"} where each
                               chosen qid appears >= 4 times; if a qid is chosen, append ALL
                               its records. Exclude any qid that already appears in prev.

        Strategy:
          1) Build qid -> [records] for sources of interest.
          2) Keep only qids with count >= 4 and not in prev.
          3) Shuffle qids and greedily add whole groups while total <= target.
          4) If still far from target, add the single group that gets total closest (allow overshoot).
        """
        # --- Paths (edit if needed) ---
        full_dataset_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
        prev_sample_path  = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval_shp")
        output_path       = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval_shp_final")

        sources_of_interest = {"shp-2-reddit", "shp-2-stackexchange"}
        target_count = 600
        seed = 42

        # --- Load inputs ---
        with full_dataset_path.open("r", encoding="utf-8") as f:
            all_records = json.load(f)
        with prev_sample_path.open("r", encoding="utf-8") as f:
            prev_records = json.load(f)

        prev_qids = {r.get("question_id") for r in prev_records if r.get("question_id")}

        # --- Build qid -> records for sources of interest ---
        qid_to_records = defaultdict(list)
        for rec in all_records:
            if rec.get("source") in sources_of_interest:
                qid = rec.get("question_id")
                if qid:
                    qid_to_records[qid].append(rec)

        # --- Candidate qids: freq >= 4 and not already in prev ---
        candidate_qids = [qid for qid, items in qid_to_records.items()
                          if len(items) >= 4 and qid not in prev_qids]

        # If nothing eligible, just copy prev and exit
        if not candidate_qids:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(prev_records, f, ensure_ascii=False, indent=2)
            print("[WARN] No eligible qids (freq>=4 & not in prev). Output is unchanged.")
            return

        # --- Shuffle and greedily pack groups up to target_count ---
        rng = random.Random(seed)
        rng.shuffle(candidate_qids)

        selected_qids = []
        total = 0

        for qid in candidate_qids:
            group_len = len(qid_to_records[qid])
            if total + group_len <= target_count:
                selected_qids.append(qid)
                total += group_len

        # Try adding one extra group (if any remain) that gets us closest to the target (may overshoot)
        remaining = [qid for qid in candidate_qids if qid not in selected_qids]
        if remaining and total < target_count:
            # pick group that minimizes |(total + group_len) - target|
            best_qid = min(remaining, key=lambda q: abs((total + len(qid_to_records[q])) - target_count))
            # only add if it improves closeness (allow tie → add)
            if abs((total + len(qid_to_records[best_qid])) - target_count) <= abs(total - target_count):
                selected_qids.append(best_qid)
                total += len(qid_to_records[best_qid])

        # --- Flatten all records for the selected qids ---
        added_records = []
        seen_ids = {r.get("question_id") for r in prev_records if r.get("question_id")}
        for qid in selected_qids:
            for rec in qid_to_records[qid]:
                # Safety: exclude any record whose qid leaked into prev (shouldn't happen due to filter)
                if rec.get("question_id") not in seen_ids:
                    added_records.append(rec)

        final = prev_records + added_records

        # --- Save ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        # --- Report ---
        num_qids_ge4 = sum(1 for q, items in qid_to_records.items() if len(items) >= 4)
        print(f"[OK] Previous size: {len(prev_records):,} (unique prev qids: {len(prev_qids):,})")
        print(f"[OK] QIDs with freq ≥ 4 in sources: {num_qids_ge4:,}")
        print(f"[OK] Candidate qids (excluding prev): {len(candidate_qids):,}")
        print(f"[OK] Selected qids: {len(selected_qids):,}")
        print(f"[OK] Appended records (ALL from chosen qids): {len(added_records):,} (target ≈ {target_count})")
        print(f"[OK] New total: {len(final):,} → {output_path}")

    @classmethod

    def filter_and_save(cls):
        input_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
        output_path = Path(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1_human_expert")

        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)   # expects JSON array

        # dictionary to keep per-source counters
        counters = defaultdict(int)
        limited = []

        for rec in records:
            if rec.get("source") in {"Chatbot Arena", "lfqa_eval"} and rec.get("human_expert") is True:
                src = rec["source"]
                if counters[src] < 260:   # only append until 260 reached
                    limited.append(rec)
                    counters[src] += 1

        # save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(limited, f, ensure_ascii=False, indent=2)

        print(f"[OK] Loaded: {len(records):,} records")
        print(f"[OK] Kept after filter + per-source limit: {len(limited):,}")
        for src, cnt in counters.items():
            print(f"    {src}: {cnt}")
        print(f"[OK] Saved to: {output_path}")
    @classmethod
    def find_duplicates_in_chatbot_arena(cls):
        # Path to JSON file
        json_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010"

        # Load JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter only Chatbot Arena records
        arena_records = [item for item in data if item.get("source") == "Chatbot Arena"]

        # Extract normalized question_text
        questions = [item.get("question_text", "").strip().lower() for item in arena_records]

        # Count occurrences
        counts = Counter(questions)

        # Find duplicates
        duplicates = {q: c for q, c in counts.items() if c > 1}

        if duplicates:
            print("Duplicated questions in Chatbot Arena:")
            for q, c in duplicates.items():
                print(f"- {q} (appears {c} times)")
            
            print("\nSummary:")
            print(f"Unique duplicated questions: {len(duplicates)}")
            print(f"Total duplicate entries: {sum(c for c in duplicates.values())}")
        else:
            print("No duplicates found in Chatbot Arena.")

    @classmethod
    def duplicate_answer_finder(cls):
        export_path = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010_duplicate_answers"
        json_path = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010"
       
        preview_chars = 120
        show_examples = 10
        data = None
        # Load
        if data is None:
            if not json_path:
                raise ValueError("Provide either `json_path` or `data`.")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Collect answers per qid (normalize inline)
        answers_per_qid = defaultdict(list)  # qid -> list of (norm, raw, rec_idx, field)
        for idx, rec in enumerate(data):
            qid = rec.get("question_id")
            if not qid:
                continue
            for k, v in rec.items():
                if isinstance(k, str) and k.lower().startswith("answer") and isinstance(v, str) and v.strip():
                    s = html.unescape(v)
                    s = re.sub(r"<[^>]+>", " ", s)      # strip HTML tags if any
                    s = re.sub(r"\s+", " ", s).strip().lower()
                    if s:
                        answers_per_qid[qid].append((s, v, idx, k))

        # Find duplicates within each qid
        overlaps_within = {}  # qid -> {norm_answer: [(idx,fld,raw), ...]} where count > 1
        for qid, items in answers_per_qid.items():
            bucket = defaultdict(list)
            for norm, raw, ridx, fld in items:
                bucket[norm].append((ridx, fld, raw))
            dup_only = {a: lst for a, lst in bucket.items() if len(lst) > 1}
            if dup_only:
                overlaps_within[qid] = dup_only

        # Report
        print("=== Intra-question_id Overlap Report ===")
        print(f"Total records: {len(data)}")
        print(f"Question IDs with any answers: {len(answers_per_qid)}")
        print(f"Question IDs with duplicate/overlapping answers: {len(overlaps_within)}\n")
        if not overlaps_within:
            print("No within-qid answer overlaps found.")
        else:
            for qid, dups in overlaps_within.items():
                print(f"- question_id={qid}: {len(dups)} duplicated answer(s)")
                shown = 0
                for _, occs in dups.items():
                    preview = re.sub(r"\s+", " ", occs[0][2]).strip()
                    if len(preview) > preview_chars:
                        preview = preview[:preview_chars] + "…"
                    where = ", ".join([f"(rec#{i}, {fld})" for i, fld, _ in occs])
                    print(f"    · '{preview}' appears {len(occs)} times at {where}")
                    shown += 1
                    if shown >= show_examples:
                        break

        if export_path:
            with open(export_path, "w", encoding="utf-8") as out:
                json.dump(overlaps_within, out, ensure_ascii=False, indent=2)

        return overlaps_within

    @classmethod
    def sample_and_save(cls):
        # Load dataset

        input_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010"
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Randomly sample 100
        sampled = random.sample(data, 100)
        output_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100"
        # Save as JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for record in sampled:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Sampled 100 records saved to {output_file}")



    @classmethod
    def check_sample_count(cls):
        # Count lines in sampled JSONL
        output_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__human_expert_chatarena_lfqa_eval_shp_final"
        with open(output_file, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"Number of records in {output_file}: {count}")






def main():
    humanfilter = HumanFilter()
    humanfilter.check_sample_count()


if __name__ == "__main__":
    main()
