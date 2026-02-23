from datetime import datetime
import hashlib
import json
import os
from pandas import concat, DataFrame, read_csv
from typing import Dict, Sequence, Tuple

from ao3_tags import DATA_PATH, RESOURCE_PATH


def clean_tag(tag):
    return tag.replace("*a*", "&").replace("*d*", ".").replace("*h*", "#").replace("*q*", "?").replace("*s*", "/")


class JobPreparation:
    """
    Class to select which tags to use in jobs such as log_ratio calculation or mannwhitneyu test.
    The central method is "prepare" that can receive category lists as parameters.
    If nothing is provided, categories are selected interactively via the command line
    """

    def __init__(self, warning: str, data_path: str = str(DATA_PATH)):
        # Load the tags and categories for the warning
        self.warning = warning
        self.tag_df = read_csv(RESOURCE_PATH / f"tags/{warning}.csv")
        self.categories = [
            x for x in self.tag_df.columns.to_list()
            if not x in ["src", "relationship", "dst"]
        ]

        # Load the JSON and CSV with job metadata
        self.job_json = data_path + f"/job_tags.json"
        if os.path.exists(self.job_json):
            with open(self.job_json, "r") as f:
                self.job_tags = json.load(f)
        else:
            self.job_tags = {}

        self.job_csv = data_path + f"/job_metadata.csv"
        if os.path.exists(self.job_csv):
            self.job_metadata = read_csv(self.job_csv)

        else:
            self.job_metadata = DataFrame(
                columns=["id", "warning", "test_categories", "excluded_categories", "created"]
            )

    def prepare(self,
                test_categories: Sequence[str] = None,
                excluded_categories: Sequence[str] = None,
                ):
        """
        Select tags for defined categories and write them to the job metadata file.
        This metadata can then be used by the spark job.
        """
        # 1. Get user input if no categories were provided
        if not test_categories:
            test_categories, excluded_categories = self.select_categories()

        # 2. Check if the combination of categories is covered by previous jobs
        matched_df = self.job_metadata.loc[
            (self.job_metadata["warning"] == self.warning) &
            (self.job_metadata["test_categories"].apply(lambda x: eval(x) == test_categories)) &
            (self.job_metadata["excluded_categories"].apply(lambda x: eval(x) == excluded_categories))
            ]

        os.system("clear")

        if len(matched_df) > 0:
            job_id = matched_df["id"].values[0]
            print("The category combination was covered by a previous job with the following id")

        else:
            # 3. Get the tags that belong to the categories
            # (excluded_tags contains original test_tags to remove them from the baseline)
            test_tags, excluded_tags = self.select_tags(test_categories=test_categories,
                                                        excluded_categories=excluded_categories)

            # 4. Write the job metadata
            datestr, job_id = self._generate_id()
            self.job_tags[job_id] = {"test_tags": [clean_tag(t) for t in test_tags],
                                     "excluded_tags": [clean_tag(t) for t in excluded_tags]}

            new_df = DataFrame([{"id": job_id,
                                 "warning": self.warning,
                                 "test_categories": test_categories,
                                 "excluded_categories": excluded_categories,
                                 "created": datestr}])
            self.job_metadata = concat([self.job_metadata, new_df], ignore_index=True)

            print("Saving updated job meta data...")
            self.job_metadata.to_csv(self.job_csv, index=False)
            with open(self.job_json, 'w') as f:
                json.dump(self.job_tags, f)

        # Print the job id to be used by the spark job
        print("\n" + "=" * 40)
        print(f"Job ID: {job_id}")
        print("=" * 40)

    def select_categories(self) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Returns two lists of categories:
        1.  The categories that should be tested for their vocabulary
        2.  The categories that should be removed from the baseline that is going to be tested against
        """
        # Get the categories to be tested and those to be excluded from the baseline that is compared against
        test_categories = self._get_category_input(question="Which categories should be tested against the remaining ones?")
        if len(test_categories) == 0:
            print("You selected no categories to be tested. Closing")
            exit(0)
        excluded_categories = self._get_category_input(question="Should any categories be excluded from the remaining ones?",
                                                  covered_categories=test_categories)
        return test_categories, excluded_categories

    def _get_category_input(self, question: str, covered_categories: Sequence[str] = None) -> Sequence[str]:
        covered_categories = covered_categories or []

        self._print_categories(covered_categories=covered_categories)
        print(f"\n{question}")
        user_input = input("Please provide a comma-separated list of category numbers (e.g., '6, 8'): ")

        try:
            idx_list = eval(f"[{user_input}]") if len(user_input) > 0 else []
        except:
            print("You must provide a comma-separated list of category numbers.")
            _ = input(f"{user_input} is invalid. Press enter to redo the entry.")
            return self._get_category_input(question=question, covered_categories=covered_categories)

        category_list = [category for i, category in enumerate(self.categories) if i in idx_list]
        if category_list:
            print(f"\nThe following categories will be selected: {', '.join(category_list)}")
        else:
            print(f"\nNo categories were selected")

        user_input = input("Please confirm or reject (y/n): ")

        if user_input.lower() == "y":
            return category_list
        else:
            return self._get_category_input(question=question, covered_categories=covered_categories)

    def _print_categories(self, covered_categories: Sequence[str] = None) -> None:
        covered_categories = covered_categories or []
        print(covered_categories)

        os.system("clear")
        print(f"Available categories for {self.warning}" + "\n" + "-" * 30)
        for i, a in enumerate(self.categories):
            if a not in covered_categories:
                print(f"{i}: {a}")

    def select_tags(self, test_categories: Sequence[str], excluded_categories: Sequence[str]) \
            -> Tuple[Sequence[str], Sequence[str]]:
        """
        Returns two lists of tags
        1.  The tags that belong to one of the categories in "test_categories".
            Chapters with these tags will be assigned to group A
        2.  The tags that belong to one of the categories in "excluded_categories".
            Chapters with these tags will not be included in the baseline that is compared against (includes test_tags)
        """
        test_tags = self.tag_df.loc[self.tag_df[test_categories].sum(axis=1) == 1]["dst"].tolist()
        excluded_tags = self.tag_df.loc[self.tag_df[excluded_categories].sum(axis=1) == 1]["dst"].tolist()

        # Expand the excluded categories with the test categories to also exclude them from the baseline
        return test_tags, list(set([*test_tags, *excluded_tags]))

    @staticmethod
    def _generate_id(hash_len: int = 10) -> Tuple[str, str]:
        """
        Get the current datetime. Return it as a formatted string together with a fixed length hash of the timestamp
        """
        dt = datetime.now()
        return (dt.strftime('%Y-%m-%d %H:%M:%S'),
                hashlib.md5(str(dt.timestamp()).encode("utf-8")).hexdigest()[:hash_len])


if __name__ == '__main__':
    # Parse the input arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to select tags")
    args = parser.parse_args()

    # Run the job as specified
    jp = JobPreparation(warning=args.warning)
    jp.prepare()
