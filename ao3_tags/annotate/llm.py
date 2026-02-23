from distutils.util import strtobool
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from typing import Sequence

from ao3_tags import MODEL_PATH
from ao3_tags.annotate.abstract import Annotator, command_line_params

class LlmAnnotator(Annotator):
    def __init__(self, warning: str, category: str, job_id: str, model_name: str = "google/flan-t5-xxl",
                 cache_dir: str = None, quantized: bool = True, n_profiles: int = 10, batch_size: int = 4,
                 sample_passages: bool = True, num_passages: int = 10_000, min_len: int = 500, max_len: int = 1_000):
        """
        Create an Annotator instance for LLM annotations and the specified job and passages
        :param warning:             Warning for which to annotate passages
        :param category:            Category of the warning. Used to construct prompts
        :param job_id:              ID of the job to run. Used to define the file
        :param n_profiles:          Number of sociodemographic profiles to use for the annotations
        :param batch_size:          Batch size to use in inference

        Parameters for the model
        :param model_name:          Name of the Hugging Face model to use
        :param cache_dir:           Directory from where to load model checkpoints
        :param quantized:           Whether to quantize the model

        Parameters for passage sampling:
        :param sample_passages:     True: Sample passages for annotation; False: Annotate all that are available
        :param num_passages:        Number of passages to sample for annotation.
                                    If passages were already sampled for the job id, they are reused
        :param min_len:             Minimum number of characters for a passage to be sampled
        :param max_len:             Maximum number of characters for a passage to be sampled
        """

        super().__init__(warning=warning, category=category, job_id=job_id, n_profiles=n_profiles,
                         batch_size=batch_size, sample_passages=sample_passages, num_passages=num_passages,
                         min_len=min_len, max_len=max_len)
        cache_dir = cache_dir or MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if quantized:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto",
                                                               quantization_config=quantization_config,
                                                               cache_dir=cache_dir, torch_dtype=torch.float16)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir,
                                                               torch_dtype=torch.float16)

    def response_func(self, prompts: Sequence[str]) -> Sequence[str]:
        responses = []
        try:
            input_ids = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).input_ids.to('cuda')
            outputs = self.model.generate(input_ids, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        except Exception as e:
            print(f"\nFailed to generate responses due to: {e}")
            exit(1)

        return responses


def run(warning: str, category: str, job_id: str, model_name: str = "google/flan-t5-xxl", cache_dir: str = None,
        quantized: bool = True, n_profiles: int = 10, batch_size: int = 4, sample_passages: bool = True,
        num_passages: int = 10_000, min_len: int = 500, max_len: int = 1_000):
    """
    Get annotations for the sampled passages using the specified model. Default cache_dir is DATA_PATH.parent / "models"
    """
    annotator = LlmAnnotator(warning=warning, category=category, job_id=job_id, model_name=model_name,
                             cache_dir=cache_dir, quantized=quantized, n_profiles=n_profiles, batch_size=batch_size,
                             sample_passages=sample_passages, num_passages=num_passages, min_len=min_len,
                             max_len=max_len)
    annotator.get_annotations()


if __name__ == "__main__":
    # Parse the input arguments
    parser = command_line_params()
    parser.add_argument("--model_name", metavar="m", type=str, default="google/flan-t5-xxl",
                        help='Name of the Hugging Face model to use')
    parser.add_argument("--cache_dir", metavar="c", type=str, default=None,
                        help='Directory from where to load model checkpoints')
    parser.add_argument("--quantized", metavar="c", type=lambda x: bool(strtobool(x)), default=True,
                        help='Whether to quantize the model')
    args = parser.parse_args()

    run(warning=args.warning, category=args.category, job_id=args.job_id, model_name=args.model_name,
        cache_dir=args.cache_dir, quantized=args.quantized, n_profiles=args.n_profiles, batch_size=args.batch_size,
        sample_passages=args.sample_passages, num_passages=args.num_passages, min_len=args.min_len,
        max_len=args.max_len)
