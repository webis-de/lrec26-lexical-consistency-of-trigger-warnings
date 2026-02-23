from ao3_tags import RESOURCE_PATH
from pandas import DataFrame, read_csv
from openai import OpenAI
import re

class VocabularySuggester:
    def __init__(self, category: str, api_key: str):
        self.category = category
        self.client = OpenAI(api_key=api_key)
        self.file_prefix = str(RESOURCE_PATH / f"categories/{category}")
        self.indicators = read_csv(self.file_prefix + "_indicators.csv")["indicator"].tolist()

    def create_word_csv(self, model: str = "gpt-4o", batch_size: int = 10):
        len_ = len(self.indicators)
        results = []
        out_file = self.file_prefix + "_words.csv"

        # Verify prompt with user
        prompt = self._create_prompt(indicators=self.indicators[0:2])
        print("\n\nThis is an example prompt for the first two indicators:\n" + "-"*50 + f"\n{prompt}\n" + "-"*50)
        user_input = input("\n\nShould this prompt be used to generate a vocabulary? (y/n): ")
        if user_input.lower() != "y":
            print("User declined prompt")
            exit(0)

        print(f"Prompting {model}...")
        for start, end in [(i, min(i+batch_size, len_)) for i in range(0, len_, batch_size)]:
            batch = self.indicators[start:end]
            prompt = self._create_prompt(indicators=batch)
            answer = self._prompt_gpt(prompt=prompt, model=model)

            try:
                results = self._parse_answer(answer=answer, results=results)
                DataFrame(results).to_csv(out_file, index=False)
                print(f"- Parsed indicators {start} to {end}")

            except Exception as e:
                self._save_failed_parsing(answer=answer, e=e, start=start, end=end)

        DataFrame(results).to_csv(out_file, index=False)
        print(f"\nSaved results to {out_file}.")

    @staticmethod
    def _parse_answer(answer, results):
        suggestions = answer.split("***")[1:]
        for indicator, wordlist in [(suggestions[i], suggestions[i+1])
                                    for i in range(0, len(suggestions) - 1, 2)]:
            words = [w.replace("\n", "").lower() for w in re.split(r"\d+. ", wordlist) if w not in ["", "\n"]]
            for w in words:
                results.append({"indicator": indicator, "word": w})

        return results

    def _save_failed_parsing(self, answer, e, start, end):
        print(f"Failed parsing the response with exception {e}.")
        print(f"\nThis is the answer by the model: \n\n{answer}\n\n")
        answer_file = self.file_prefix + f"_answer_{start}-{end}.txt"
        with open(answer_file, "w") as file:
            file.write(answer)
        print(f"The answer was saved to {answer_file}.")

        user_input = input("Do you wish to continue with the next batch of indicators? (y/n) ")
        if user_input.lower() != "y":
            exit(0)

    def _create_prompt(self, indicators):
        indicator_list = '\n'.join(indicators)
        category = str(self.category).replace("_", " ")
        prompt = f"We want to automatically detect text passages in stories related to {category}. " \
                 f"Experts collected the following indicators: \n\n{indicator_list}\n\n" \
                 "Please suggest 20 individual words (nouns, verbs or adjectives) for each indicator " \
                 f"that relate to {category}. " \
                 f"Please apply the following rules: \n" \
                 "1. Ensure a mixture of nouns, verbs and adjectives\n" \
                 "2. Return only the suggestions as a list, separated by '***[INDICATOR]***'.\n\n" \
                 "Example:" \
                 f"\n\n***{indicators[0]}***" \
                 "\n1. Word A" \
                 "\n2. Word B" \
                 "\n..." \
                 f"\n***NEXT INDICATOR***\n..."
        return prompt

    def _prompt_gpt(self, prompt: str, model: str) -> str:
        completion = self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model)
        return completion.choices[0].message.content


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='GPT4 Suggestion',
        description='Send prompt to GPT-4 to suggest words related to an category of a warning. '
                    'This script requires a manually curated CSV file of indicators for the category.')

    parser.add_argument("category", metavar="C", type=str,
                        help="Category to collect words for. Example: 'physical_abuse'")
    parser.add_argument("key", metavar="K", type=str, help="Key for the OpenAI API")
    args = parser.parse_args()

    suggester = VocabularySuggester(category=args.category, api_key=args.key)
    suggester.create_word_csv()