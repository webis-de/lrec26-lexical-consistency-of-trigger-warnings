import click
import krippendorff
import pandas as pd

from ao3_tags import VISUALIZATION_PATH
from ao3_tags.stats.annotate.utils import load_annotations
from ao3_tags.visualization import ID_MAP


CATEGORY_TO_ID = {
    v: k for k, v in ID_MAP.items()
}
SD_COLUMNS = [
    "gender",
    "race",
    "education",
    "age",
    "political_affiliation",
]

@click.command("Create a heatmap of Krippendorff's alpha values for groupwise agreement")
@click.option(
    "--warning",
    prompt="Name of the high-level warning",
    type=str,
    default="abuse")
def main(warning: str):
    df = pd.DataFrame()

    for category in CATEGORY_TO_ID.keys():
        df_alpha = category_alpha(warning=warning, category=category)
        df_alpha["category"] = category
        df = pd.concat([df, df_alpha])

    df = df.pivot(index=["sd_attribute", "sd_value"], columns="category", values="alpha").reset_index()
    output_path = VISUALIZATION_PATH / f"group_agreement.csv"
    df.to_csv(output_path, index=False)
    print(f"\n\nCreated CSV of groupwise Krippendorff's Alpha under {output_path}")


def category_alpha(warning: str, category: str):
    print(f"\nKrippendorffs's Alpha for {category}")
    job_id = CATEGORY_TO_ID[category]
    df = load_annotations(warning=warning, job_id=job_id)

    records = []
    for col in SD_COLUMNS:
        for unique_v in df[col].unique():
            df_group = df[df[col] == unique_v]
            if len(df_group["annotator_id"].unique()) < 2:
                print(f"    * Value '{unique_v}' for column '{col}' has less than 2 associated profiles. It will be skipped.")
                continue

            alpha = calculate_alpha_group(df_group)
            records.append({
                "sd_attribute": col,
                "sd_value": unique_v,
                "alpha": alpha,
            })

    print("- Collected all groupwise agreements")
    return pd.DataFrame(records)


def calculate_alpha_group(df_group: pd.DataFrame) -> float:
    unique_profiles = df_group["annotator_id"].unique()

    reliability_data = []
    passage_ids = None

    for id_ in unique_profiles:
        df_profile = df_group[df_group["annotator_id"] == id_].sort_values("passage_id")
        if passage_ids is None:
            passage_ids = df_profile["passage_id"].to_list()

        assert df_profile["passage_id"].to_list() == passage_ids, f"Annotator {id_} has different passage_ids"
        reliability_data.append(df_profile["response"].to_list())

    return krippendorff.alpha(reliability_data)



if __name__ == "__main__":
    main()