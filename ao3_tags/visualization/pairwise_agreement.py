import click
import itertools
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import krippendorff
import pandas as pd
import seaborn as sns

from ao3_tags import VISUALIZATION_PATH
from ao3_tags.stats.annotate.utils import load_annotations
from ao3_tags.utils import capitalize_category
from ao3_tags.visualization import ID_MAP, COLOR_MAP

CATEGORY_TO_ID = {
    v: k for k, v in ID_MAP.items()
}
HEATMAP_SCALE = [0.6, 0.9]

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.size'] = 13


@click.command("Create a heatmap of Krippendorff's alpha values for pairwise agreement")
@click.option(
    "--warning",
    prompt="Name of the high-level warning",
    type=str,
    default="abuse")
@click.option(
    "--category",
    prompt="Category within the warning",
    type=str,
    default="emotional_abuse"
)
def main(warning: str, category: str):
    job_id = CATEGORY_TO_ID[category]
    df_heatmap = calculate_alpha(warning=warning, job_id=job_id)

    cap_category = capitalize_category(category)

    # Create the colormap and plot the heatmap
    cmap = LinearSegmentedColormap.from_list(name=category, colors=list(zip(
        [0, 1],
        ["#FFFFFF", COLOR_MAP[category]]
    )))
    ax = sns.heatmap(data=df_heatmap,
                cmap=cmap,
                annot=True,
                fmt=".2f",
                linewidth=.5,
                vmin=HEATMAP_SCALE[0],
                vmax=HEATMAP_SCALE[1])

    ax.set(xlabel="Profile", ylabel="Profile")
    ax.set_title(f"{cap_category} [Krippendorff's $\\alpha$]", x=0.5, y=1.02)
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = VISUALIZATION_PATH / f"pairwise_agreement_{category}.pdf"
    plt.savefig(output_path, dpi=500)
    print(f"- Created visualization of Krippendorff's Alpha ({cap_category}) under {output_path}")


def calculate_alpha(warning: str, job_id: str):
    df = load_annotations(warning=warning, job_id=job_id)
    df["annotator_id"] = df["annotator_id"] + 1

    ids = list(range(1, df["annotator_id"].max() + 1))

    print("- Calculating Krippendorffs's Alpha")
    records = []
    for a, b in itertools.combinations(ids, 2):
        df_a = df[df["annotator_id"] == a].sort_values("passage_id")
        df_b = df[df["annotator_id"] == b].sort_values("passage_id")
        assert df_a["passage_id"].to_list() == df_b[
            "passage_id"].to_list(), f"Annotator {a} and {b} have different passage_ids"

        alpha = krippendorff.alpha([df_a["response"].to_list(), df_b["response"].to_list()])
        records.append({
            "ID_A": a,
            "ID_B": b,
            "alpha": alpha
        })

    df_kappa = pd.DataFrame(records)
    return df_kappa.pivot(columns="ID_B", index="ID_A", values="alpha")


if __name__ == "__main__":
    main()