import json
import random
from collections import defaultdict, Counter
from pathlib import Path

def run_domain_aware_sampling():
    class DomainAwareSampler:
        def __init__(self):
            # === Edit these two paths if needed ===
            self.input_file  = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010"
            self.output_file = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100"
            self.seed = 42

            # Fixed source names (as confirmed)
            self.SOURCES_FIXED = ("Chatbot Arena", "lfqa_eval", "shp-2-reddit", "shp-2-stackexchange")

        def load(self):
            with open(self.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Keep only records with known sources
            return [r for r in data if r.get("source") in self.SOURCES_FIXED]

        def sample_fixed(self, records, source_name, k):
            pool = [r for r in records if r.get("source") == source_name]
            random.shuffle(pool)
            return pool[: min(k, len(pool))]

        def sample_one_per_domain_from_shp(self, records):
            shp_pool = [r for r in records if r.get("source") in ("shp-2-reddit", "shp-2-stackexchange")]
            by_domain = defaultdict(list)
            for r in shp_pool:
                d = r.get("domain")
                if d:  # require non-empty domain
                    by_domain[d].append(r)

            selected = []
            for d, items in by_domain.items():
                random.shuffle(items)
                selected.append(items[0])  # pick 1 per domain

            return selected, by_domain

        def dedup_by_qid(self, items):
            seen = set()
            out = []
            for r in items:
                qid = r.get("question_id")
                if qid and qid not in seen:
                    seen.add(qid)
                    out.append(r)
            return out

        def run(self):
            random.seed(self.seed)
            data = self.load()

            pick_arena = self.sample_fixed(data, "Chatbot Arena", 20)
            pick_lfqa  = self.sample_fixed(data, "lfqa_eval", 20)

            pick_domains, by_domain = self.sample_one_per_domain_from_shp(data)

            sampled = pick_arena + pick_lfqa + pick_domains
            sampled = self.dedup_by_qid(sampled)

            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(sampled, f, ensure_ascii=False, indent=2)

            # Summary
            print("=== Sampling Summary ===")
            print(f"Input records (kept sources): {len(data):,}")
            print(f"Picked Chatbot Arena:         {len(pick_arena)}")
            print(f"Picked lfqa_eval:             {len(pick_lfqa)}")
            print(f"SHP domains found:            {len(by_domain)}")
            print(f"Picked 1 per domain:          {len(pick_domains)}")
            print(f"TOTAL sampled (deduped):      {len(sampled)}")
            print(f"Saved to:                     {self.output_file}")

            src_counts = Counter(r.get("source") for r in sampled)
            print("\nBy source in sample:")
            for s, c in sorted(src_counts.items()):
                print(f"  {s:<20} {c}")

    DomainAwareSampler().run()

def main():
    run_domain_aware_sampling()


if __name__ == "__main__":
    main()