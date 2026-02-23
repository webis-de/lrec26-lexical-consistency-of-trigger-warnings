import json
import pandas as pd

from ao3_tags import DATA_PATH


def load_annotations(warning: str, job_id: str):
    print("- Reading annotations...")
    annotation_dir = DATA_PATH / "annotations" / warning
    with open(annotation_dir / f"{job_id}_annotations.jsonl", "r") as f:
        results = [json.loads(l) for l in f.readlines()]
        df = pd.DataFrame(results)
    df["response"] = df["response"].apply(
        lambda x: 1 if x.lower() == "yes" else (0 if x.lower() == "no" else None)
    )
    return df
