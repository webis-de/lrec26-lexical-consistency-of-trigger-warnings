import requests
from resiliparse.parse.html import *
import pandas as pd

from ao3_tags import DATA_PATH


def get_tags(ao3_url: str = "https://archiveofourown.org/tags/Violence/"):
    res = requests.get(ao3_url)
    tree = HTMLTree.parse(res.text)
    tags = tree.body.query_selector(".synonym.listbox.group") \
                    .query_selector(".tags.commas.index.group") \
                    .query_selector_all("li")
    return [tag.text for tag in tags]


def get_tag_list(warning: str = "violence"):
    url_str = f"https://archiveofourown.org/tags/{warning}/"
    try:
        tags = get_tags(url_str)
    except:
        print(f"Unable to retrieve tags for {url_str}. Please verify the URL.")
        tags = []
    return tags


def save_tag_csv(warning: str = "violence"):
    (DATA_PATH / "tags").mkdir(exist_ok=True)
    out_path = DATA_PATH / "tags" / f"{warning}_ao3_tag_page.csv"
    tags = get_tag_list(warning)
    data = {"original_tag": tags, "notes": [None for t in tags]}
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"\nCollected tags for {warning}. Saved under {out_path}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str,
                        help="Tag (e.g. 'Violence') for which to collect synonyms (e.g. 'Warning for Violence')")

    args = parser.parse_args()
    save_tag_csv(args.warning)
