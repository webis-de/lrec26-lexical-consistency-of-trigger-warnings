import requests
import time
from typing import Sequence
import yaml

from ao3_tags import CONFIG_PATH
from ao3_tags.annotate.abstract import Annotator, command_line_params

with open(CONFIG_PATH / "config.yaml") as f:
    conf_dict = yaml.safe_load(f)
    API_KEY = (conf_dict["huggingface"]["key"])

headers = {"Authorization": f"Bearer {API_KEY}"}
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


class ApiAnnotator(Annotator):
    def __init__(self, warning: str, category: str, job_id: str, req_per_min: int = 1_000,
                 max_retries: int = 3, n_profiles: int = 10, batch_size: int = 4, sample_passages: bool = True,
                 num_passages: int = 10_000, min_len: int = 500, max_len: int = 1_000):
        """
        Create an Annotator instance for API annotations and the specified job and passages
        :param warning:             Warning for which to annotate passages
        :param category:            Category of the warning. Used to construct prompts
        :param job_id:              ID of the job to run. Used to define the file
        :param n_profiles:          Number of sociodemographic profiles to use for the annotations
        :param batch_size:          Batch size to use in inference

        Parameters for the API
        :param req_per_min:         Number of requests made per minute
        :param max_retries:         Maximum number of retries after to abort the job

        Parameters for passage sampling:
        :param sample_passages:     True: Sample passages for annotation; False: Annotate all that are available
        :param num_passages:        Number of passages to sample for annotation.
                                    If passages were already sampled for the job id, they are reused
        :param min_len:             Minimum number of characters for a passage to be sampled
        :param max_len:             Maximum number of characters for a passage to be sampled
        """
        super().__init__(warning=warning, category=category, job_id=job_id, n_profiles=n_profiles,
                         batch_size=batch_size, sample_passages=sample_passages,
                         num_passages=num_passages, min_len=min_len, max_len=max_len)
        self.req_per_min = req_per_min
        self.max_retries = max_retries

    def response_func(self, prompts: Sequence[str]) -> Sequence[str]:
        sleep_time = 60 / self.req_per_min
        responses = []

        try:
            for p in prompts:
                i = 1
                data = query(payload={"inputs": p})
                while isinstance(data, dict) and "error" in data.keys():
                    print(data)
                    print(f"Received error - Try #{i}")
                    time.sleep(60)
                    data = query(payload={"inputs": p})
                    i += 1
                    if i >= self.max_retries:
                        print("\n" * 5 + "Max retries exceeded" + "\n" * 5)
                        exit(1)

                responses.append(data[0]["generated_text"])
                time.sleep(sleep_time)

        except Exception as e:
            print(f"\nFailed to generate responses due to: {e}")
            exit(1)

        return responses


def run(warning: str, category: str, job_id: str, req_per_min: int = 1_000, max_retries: int = 3, n_profiles: int = 10,
        batch_size: int = 4, sample_passages: bool = True, num_passages: int = 10_000, min_len: int = 500,
        max_len: int = 1_000):
    """
    Get annotations for the sampled passages using the Hugging Face API.
    The req_per_min is used to regulate the requests per minute made to the Hugging Face API
    """
    annotator = ApiAnnotator(warning=warning, category=category, job_id=job_id, req_per_min=req_per_min,
                             max_retries=max_retries, n_profiles=n_profiles,
                             batch_size=batch_size, sample_passages=sample_passages, num_passages=num_passages,
                             min_len=min_len, max_len=max_len)
    annotator.get_annotations()


if __name__ == "__main__":
    parser = command_line_params()
    parser.add_argument("--req_per_min", metavar="r", type=int, default=1_000,
                        help='Number of requests made per minute')
    parser.add_argument("--max_retries", metavar="m", type=int, default=3,
                        help='Maximum number of retries after to abort the job')
    args = parser.parse_args()

    run(warning=args.warning, category=args.category, job_id=args.job_id, req_per_min=args.req_per_min,
        max_retries=args.max_retries, n_profiles=args.n_profiles, batch_size=args.batch_size,
        sample_passages=args.sample_passages, num_passages=args.num_passages, min_len=args.min_len,
        max_len=args.max_len)
