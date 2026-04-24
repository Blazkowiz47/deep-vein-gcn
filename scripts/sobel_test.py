import json
from scipy import stats
from pathlib import Path


def load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as fp:
        records = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["key"]] = record["value"]
        return records


component_ablation_path = "ablation/ablation_component_eers.jsonl"
loss_ablation_path = "ablation/ablation_input_loss_eers.jsonl"


def groupify(data: dict) -> dict:
    result = {}
    for key, value in data.items():
        if ":none" in key:
            key = key.replace(":none", "")
        _, seed, key = key.split(":")
        result.setdefault(key, {})[seed] = value
    return result


def get_avg_ci(data):
    avg = sum(data) / len(data)
    ci = stats.t.interval(0.95, len(data) - 1, loc=avg, scale=stats.sem(data))
    return avg, ci


if __name__ == "__main__":
    print("Proposed: 7.53 [5.88 - 9.18]")
    print()
    res = groupify(load_jsonl(Path(loss_ablation_path)))
    print("Loss ablation: ")
    for key, data in res.items():
        avg, ci = get_avg_ci(list(data.values()))
        print(f"{key}: {avg:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]")

    res = groupify(load_jsonl(Path(component_ablation_path)))
    print("Component ablation: ")
    for key, data in res.items():
        avg, ci = get_avg_ci(list(data.values()))
        print(f"{key}: {avg:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]")
