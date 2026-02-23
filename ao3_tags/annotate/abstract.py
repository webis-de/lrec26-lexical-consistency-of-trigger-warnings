from abc import ABC, abstractmethod
import argparse
from distutils.util import strtobool
import itertools
import json
import os
from tqdm import tqdm
from typing import Sequence, Tuple

from ao3_tags import DATA_PATH
from ao3_tags.annotate.utils import PromptConstructor, get_profiles
from ao3_tags.annotate.load_passages import load_passages, load_sampled_passages


# Abstract class for annotation jobs
class Annotator(ABC):
    def __init__(self, warning: str, category: str, job_id: str, n_profiles: int = 10, batch_size: int = 4,
                 sample_passages: bool = True, num_passages: int = 10_000, min_len: int = 500, max_len: int = 1_000):
        """
        Abstract Annotator class
        :param warning:             Warning for which to annotate passages
        :param category:            Category of the warning. Used to construct prompts
        :param job_id:              ID of the job to run. Used to define the file
        :param n_profiles:          Number of sociodemographic profiles to use for the annotations
        :param batch_size:          Batch size to use in inference

        Parameters for passage sampling:
        :param sample_passages:     True: Sample passages for annotation; False: Annotate all that are available
        :param num_passages:        Number of passages to sample for annotation.
                                    If passages were already sampled for the job id, they are reused
        :param min_len:             Minimum number of characters for a passage to be sampled
        :param max_len:             Maximum number of characters for a passage to be sampled
        """
        self.warning = warning
        self.category = category
        self.job_id = job_id
        self.n_profiles = n_profiles
        self.batch_size = batch_size

        # Create a prompt constructor
        self.prompt_constructor = PromptConstructor(warning=warning)
        os.makedirs(DATA_PATH / "annotations" / warning, exist_ok=True)
        self.annotations_path = DATA_PATH / "annotations" / warning / f"{job_id}_annotations.jsonl"
        print(f"- Annotations will be saved to {self.annotations_path}")

        self.sample_passages = sample_passages
        self.num_passages = num_passages
        self.min_len = min_len
        self.max_len = max_len

    def prepare_job(self) -> Tuple[itertools.product, tqdm]:
        self._load_inputs()

        # Checkpointing: Skip any combinations that are already covered
        # Ideally uses the same batch_size between jsonl file and new job
        total_combinations = len(self.df_inputs) * len(self.profiles)
        print("\n")
        pbar = tqdm(total=total_combinations, desc="Getting annotations")
        num_skips = self._skip_covered_combinations(pbar=pbar)
        if num_skips*self.batch_size >= total_combinations:
            print(f"Annotations for all combinations were already recorded in {self.annotations_path}.")
            exit(1)

        # Create the input generator and loop in steps of batch_size; Print update with progress bar
        combinations = itertools.product(self.df_inputs, self.profiles)

        # Skip any inputs that were already covered
        for i in range(num_skips):
            _ = self.create_batch_input(combinations=combinations)

        return combinations, pbar

    def create_batch_input(self, combinations: itertools.product) \
            -> Tuple[Sequence[str], Sequence[str], Sequence[str], Sequence[str], Sequence[str], Sequence[str],
                     Sequence[str], Sequence[int]]:
        """
        Function to take in an iterable of combinations and construct a batch of input lists
        """
        # 1. Extract the lists of values
        df_data, profiles = zip(*take(self.batch_size, combinations))
        pass_ids, texts = zip(*df_data)

        # Turn the profiles (list of dicts) into lists for each attribute
        attr_lists = {attr: [p[attr] for p in profiles] for attr in profiles[0]}
        genders, races, edus, ages, pols, annotator_ids = (
            attr_lists["gender"], attr_lists["race"], attr_lists["education"], attr_lists["age"],
            attr_lists["political_affiliation"], attr_lists["id"]
        )

        # 2. Use the prompt constructor to get the prompts
        prompts = list(map(
            lambda profile, text:
            self.prompt_constructor.construct_prompt(profile=profile, category=self.category, text=text),
            profiles, texts
        ))

        return list(pass_ids), prompts, genders, races, edus, ages, pols, annotator_ids

    def get_annotations(self):
        """
        Central function of the Annotator class. Uses the response_func to turn a batch of prompts into responses.
        The specific response func is implemented by derived classes.
        """
        # Get the iterator for input combinations and the progress bar
        combinations, pbar = self.prepare_job()

        while True:
            # Get the next input batch
            try:
                pass_ids, prompts, genders, races, edus, ages, pols, ann_ids = \
                    self.create_batch_input(combinations=combinations)
            except Exception as e:
                print(f"Failed reading new inputs with Exception {e}")
                break

            # Obtain responses and write the output
            responses = self.response_func(prompts=prompts)
            if responses:
                self._write_results(pass_ids=pass_ids, responses=responses, genders=genders, races=races,
                                    edus=edus, ages=ages, pols=pols, ann_ids=ann_ids)
                pbar.update(self.batch_size)

        pbar.close()

    @abstractmethod
    def response_func(self, prompts: Sequence[str]) -> Sequence[str]:
        """
        Turns a list of prompts (one batch) into responses
        """
        pass

    def _write_results(self, pass_ids: Sequence[str], responses: Sequence[str], genders: Sequence[str],
                       races: Sequence[str], edus: Sequence[str], ages: Sequence[str], pols: Sequence[str],
                       ann_ids: Sequence[int]):

        with open(self.annotations_path, "a") as file:
            for tup in zip(pass_ids, responses, genders, races, edus, ages, pols, ann_ids):
                output = {
                    "passage_id": tup[0],
                    "response": tup[1],
                    "gender": tup[2],
                    "race": tup[3],
                    "education": tup[4],
                    "age": tup[5],
                    "political_affiliation": tup[6],
                    "annotator_id": tup[7]
                }
                json.dump(output, file)
                file.write("\n")

    # Checkpointing (skip combinations for which there already exist an output)
    def _skip_covered_combinations(self, pbar: tqdm) -> int:
        """
        Check all results in the annotations_path in steps of batch_size and skip combinations that were already covered.
        Return the index (in steps of batch_size) from where to resume annotations
        """

        if not self.annotations_path.is_file():
            return 0

        # Create the two generators (from combinations to process and from the file)
        combinations = itertools.product(self.df_inputs, self.profiles)

        f = open(self.annotations_path, "r")
        generator = read_in_batches(f, self.batch_size)
        # Loop over all rows in the JSONL file and find the last non-consecutive mismatch between combinations
        # and the JSONL file
        i = 0
        mismatches = []
        while True:
            # Get the next batch to be used in prompting
            try:
                pass_ids, prompts, genders, races, edus, ages, pols, ann_ids = \
                    self.create_batch_input(combinations=combinations)
            except:
                break

            # Load the next covered batch (if it fails, all previous batches were covered)
            try:
                data = [x[0] for x in take(self.batch_size, generator)]
                pass_ids_c, ann_ids_c = zip(
                    *[(x["passage_id"], x["annotator_id"]) for x in data]
                )
            except:
                break

            # Check if the new batch is already covered; If not, record a mismatch
            if not (list(pass_ids_c) == pass_ids and list(ann_ids_c) == ann_ids):
                mismatches.append(i)
            i += 1
            pbar.update(self.batch_size)

        # Add the last index of the iteration in case everything matched
        mismatches.append(i)
        f.close()

        # Return the last non-consecutive mismatch as starting point
        return [x for i, x in enumerate(mismatches) if mismatches[i - 1] != x - 1][-1]

    def _load_inputs(self):
        # Load passages by either taking previously sampled IDs or sampling new ones
        if self.sample_passages:
            df = load_sampled_passages(warning=self.warning, job_id=self.job_id, num_passages=self.num_passages,
                                       min_len=self.min_len, max_len=self.max_len)
        else:
            df = load_passages(warning=self.warning, job_id=self.job_id)

        pass_ids = df["id"].to_list()
        texts = df["passage"].to_list()
        self.df_inputs = [tup for tup in zip(pass_ids, texts)]
        self.profiles = get_profiles(n=self.n_profiles)


def take(batch_size, iterable):
    return list((itertools.islice(iterable, batch_size)))


def read_in_batches(file_object, batch_size: int):
    while True:
        data = [json.loads(jline) for jline in file_object.readlines(batch_size)]
        if not data:
            break
        yield data


def command_line_params():
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str,
                        help='Warning for which to annotate passages')
    parser.add_argument("category", metavar="c", type=str,
                        help='Category of the warning. Used to construct prompts')
    parser.add_argument("job_id", metavar="j", type=str,
                        help='ID of the job to run. Used to define the file')
    parser.add_argument("--n_profiles", metavar="n", type=int, default=10,
                        help='Number of sociodemographic profiles to use for the annotations')
    parser.add_argument("--batch_size", metavar="b", type=int, default=4,
                        help='Batch size to use in inference')
    parser.add_argument("--sample_passages", metavar="s", type=lambda x: bool(strtobool(x)), default=True,
                        help='True: Sample passages for annotation; False: Annotate all that are available')
    parser.add_argument("--num_passages", metavar="p", type=int, default=10_000,
                        help='Number of passages to sample for annotation')
    parser.add_argument("--min_len", type=int, default=500,
                        help='Minimum number of characters for a passage to be sampled')
    parser.add_argument("--max_len", type=int, default=1_000,
                        help='Maximum number of characters for a passage to be sampled')
    return parser