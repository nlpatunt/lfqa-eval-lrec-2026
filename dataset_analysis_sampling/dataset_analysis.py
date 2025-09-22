import json
from collections import Counter, defaultdict

def analyze_data(json_path: str):
    """
    Load JSON records from file and count distribution by:
      - source
      - human_expert (expert_judgment)
      - domain
      - (domain, source)
      - (domain, human_expert)
      - number of distinct domains per source
      - list of domain names per source
    """

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)  # expects JSON array
        except json.JSONDecodeError:
            # fallback: JSON Lines
            records = [json.loads(line) for line in f if line.strip()]

    # Initialize counters
    source_counts = Counter()
    expert_counts = Counter()
    domain_counts = Counter()
    domain_source_counts = Counter()
    domain_expert_counts = Counter()
    domains_per_source = defaultdict(set)  # source → set of domains

    # Count occurrences
    for rec in records:
        src = rec.get("source")
        exp = rec.get("human_expert")
        dom = rec.get("domain")

        source_counts[src] += 1
        expert_counts[exp] += 1
        domain_counts[dom] += 1
        domain_source_counts[(dom, src)] += 1
        domain_expert_counts[(dom, exp)] += 1
        if src and dom:
            domains_per_source[src].add(dom)

    # Compute number + list of domains per source
    domain_count_per_source = {src: len(doms) for src, doms in domains_per_source.items()}
    domain_list_per_source = {src: sorted(list(doms)) for src, doms in domains_per_source.items()}

    # Print summary
    print("\nCounts by source:")
    for k, v in source_counts.items():
        print(f"  {k}: {v}")

    print("\nCounts by expert_judgment (human_expert):")
    for k, v in expert_counts.items():
        print(f"  {k}: {v}")

    print("\nCounts by domain:")
    for k, v in domain_counts.items():
        print(f"  {k}: {v}")

    print("\nCounts by (domain, source):")
    for k, v in domain_source_counts.items():
        print(f"  {k}: {v}")

    print("\nCounts by (domain, expert_judgment):")
    for k, v in domain_expert_counts.items():
        print(f"  {k}: {v}")

    print("\nNumber of distinct domains per source:")
    for k, v in domain_count_per_source.items():
        print(f"  {k}: {v}")

    print("\nDomain names per source:")
    for k, v in domain_list_per_source.items():
        print(f"  {k}: {v}")

    # Return raw results too
    return {
        "by_source": dict(source_counts),
        "by_expert_judgment": dict(expert_counts),
        "by_domain": dict(domain_counts),
        "by_domain_source": dict(domain_source_counts),
        "by_domain_expert": dict(domain_expert_counts),
        "num_domains_per_source": domain_count_per_source,
        "domain_names_per_source": domain_list_per_source,
    }

def count_human_expert_answers(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # assuming file is a JSON array, not JSONL

    valid = {"answer_1", "answer_2", "tie"}
    counts = Counter(obj.get("human_judgment") for obj in data)

    # Count only the expected ones
    total_valid = sum(counts[val] for val in valid if val in counts)
    print(f"Total objects with human_judgment in {valid}: {total_valid}")

    # Print unexpected values if any
    unexpected = {k: v for k, v in counts.items() if k not in valid}
    if unexpected:
        print("\nUnexpected human_judgment values found:")
        for k, v in unexpected.items():
            print(f" - {k!r}: {v} occurrences")

    return total_valid
# ---------- Example usage ----------
if __name__ == "__main__":
    #result = analyze_data(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_10010")
    #print("\nSummary dict:", result)

    count_human_expert_answers(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1")
