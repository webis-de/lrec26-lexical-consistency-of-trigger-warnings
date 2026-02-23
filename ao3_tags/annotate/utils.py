import json
import pandas as pd
import random
from typing import Dict, Sequence, Tuple

from ao3_tags import RESOURCE_PATH

PROFILE_PATH = RESOURCE_PATH / "annotate" / "sociodemographic_profiles.json"


class PromptConstructor:
    def __init__(self, warning: str = None):
        self.warning = warning

        # Load the dataframe of prompts and filter it based on the provided warning
        self.prompt_df = pd.read_csv(RESOURCE_PATH / "annotate" / "prompts.csv")
        if warning:
            self.prompt_df = self.prompt_df.loc[self.prompt_df["warning"] == warning]

    def construct_prompt(self, profile: Dict[str, str], category: str, text: str) -> str:
        # Create the description of the sociodemographic profile
        sd_desc = self._create_sd_desc(**profile)

        # Get the correct prompt based on warning and category
        instruction, question = self.prompt_df.loc[
            (self.prompt_df["warning"] == self.warning) &
            (self.prompt_df["category"] == category)
            ][["instruction", "question"]].iloc[0]

        # Join everything together for the prompt
        prompt = f"{instruction}\n\nText: ’{text}’\n\n{question}"
        return prompt.replace("{sd_desc}", sd_desc)

    @staticmethod
    def _create_sd_desc(gender: str, race: str, education: str, age: str, political_affiliation: str, id: str = None) \
            -> str:
        """
        Create a sociodemographic description based on the provided attributes.
        The id_ is taken for compatibility with the profile dictionary.
        """
        return (f"gender ’{gender}’, "f"race ’{race}’, "f"age ’{age}’, "f"education level ’{education}’ "
                f"and political affiliation ’{political_affiliation}’")


# Create sociodemographic profiles
# Taken from [Beck et. al 2024](https://aclanthology.org/2024.eacl-long.159)
# Code: https://github.com/UKPLab/arxiv2023-sociodemographic-prompting
# Expanded with distribution of attributes used in the paper
sociodemographic_attribute_values = {
    'gender': {'male': 47.35, 'female': 52.18, 'nonbinary': 0.47},
    'race': {'Black or African American': 13.12, 'White': 76.90, 'Asian': 6.15,
             'American Indian or Alaska Native': 0.80, 'Hispanic': 2.78,
             'Native Hawaiian or Pacific Islander': 0.24},
    'education': {'Some college but no degree': 19.18,
                  'Associate degree in college (2-year)': 10.93,
                  "Bachelor's degree in college (4-year)": 41.87, 'Doctoral degree': 1.32,
                  "Master's degree": 15.90, 'Professional degree (JD, MD)': 1.59,
                  'High school graduate (high school diploma or equivalent including GED)': 8.69,
                  'Less than high school degree': 0.53},
    'age': {'35 - 44': 25.03, '18 - 24': 10.73, '25 - 34': 39.62, '45 - 54': 13.49, '55 - 64': 7.70,
            '65 or older': 3.41, 'Under 18': 0.02},
    'political_affiliation': {'Liberal': 43.04, 'Independent': 28.18, 'Conservative': 28.77}
}


def get_profiles(n: int = 10) -> Sequence[Dict[str, str]]:
    # Load existing profiles if they exist
    profile_json = {}
    if PROFILE_PATH.is_file():
        with open(PROFILE_PATH, 'r') as f:
            profile_json = json.load(f)
    if str(n) in profile_json:
        return profile_json[str(n)]

    # Otherwise, create new profiles
    # Create lists of shuffled attribute values
    attribute_lists = []
    for attribute in sociodemographic_attribute_values.keys():
        attr_values = sample_attribute(attribute=attribute, n=n)
        random.shuffle(attr_values)
        attribute_lists.append(attr_values)

    # Add a number to each profile
    attribute_lists.append([i for i in range(n)])

    # Turn the lists into sociodemographic profiles of five values
    profiles = [tup for tup in zip(*attribute_lists)]

    # Enforce constraints on the randomly generated profiles
    reordered = True
    while reordered:
        profiles, reordered = _enforce_constraints(profiles)

    # Turn the profiles into dictionaries and save them
    profiles = [
        {"gender": x[0], "race": x[1], "education": x[2], "age": x[3], "political_affiliation": x[4], "id": x[5]}
        for x in profiles
    ]
    profile_json[n] = profiles
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile_json, f)
    return profiles


def sample_attribute(attribute: str = "gender", n: int = 50) -> Sequence[str]:
    """
    Function to create a list of attribute values of length n that tries to approximate the attribute distribution
    """
    d = sociodemographic_attribute_values[attribute]

    # 1. Sample profiles according to the target distribution
    attr_values = []
    for k, share in d.items():
        num_people = int(share * n / 100)
        attr_values += [k for i in range(num_people)]

    # 2. If the list is not complete, try to fill it with missing attribute values
    len_ = len(attr_values)
    if len_ < n:
        missing = {k: v for k, v in d.items() if k not in attr_values}
        sum_ = sum([v for v in missing.values()])
        for k, v in missing.items():
            num_people = int((n - len_) * v / sum_)
            attr_values += [k for i in range(num_people)]

    # 3. If the list is still incomplete, sample by biggest differences to target distribution
    len_ = len(attr_values)
    while len_ < n:
        attr_values.append(_get_underrepresented(d, attr_values))
        len_ = len(attr_values)

    return attr_values


def _get_underrepresented(d: Dict[str, float], attr_values: Sequence[str]):
    """
    Return the key for the attribute value whose share in the sample deviates most from the distribution
    """
    ratios = {}
    for k, target in d.items():
        share = len([x for x in attr_values if x == k]) / len(attr_values)
        ratios[k] = share / target
    return min(ratios, key=ratios.get)


def _enforce_constraints(profiles: Sequence[Tuple[str, str, str, str, str]]):
    """
    Function to reorder randomly generated profiles to adhere to constraints
    """
    reordered_profiles = False
    for i in range(len(profiles)):
        education, age = profiles[i][2], profiles[i][3]
        if education in ['Associate degree in college (2-year)', "Bachelor's degree in college (4-year)",
                         "Master's degree", 'Professional degree (JD, MD)', 'Doctoral degree']:
            swap_idx = None
            if age == "Under 18":
                swap_idx = _find_swap_index(profiles, i)
            if age == "18 - 24" and education in ['Professional degree (JD, MD)', 'Doctoral degree']:
                swap_idx = _find_swap_index(profiles, i)

            if swap_idx is not None:
                profiles[i] = profiles[i][:3] + (profiles[swap_idx][3],) + profiles[i][4:]
                profiles[swap_idx] = profiles[swap_idx][:3] + (age,) + profiles[swap_idx][4:]
                reordered_profiles = True

    return profiles, reordered_profiles


def _find_swap_index(profiles: Sequence[Tuple[str, str, str, str, str]], current_idx: int):
    """
    Function used in _enforce_constraints() to find the index of a tuple to swap with
    """
    idx = None
    while idx == current_idx or idx is None:
        idx = random.randint(0, (len(profiles) - 1))
    return idx
