# Create a category vocabulary
## Create an indicator file
The first step to creating a vocabulary is creating a file of indicators in [resources/categories](../../resources/categories).
These indicators are ideally derived from authoritative sources like the examples for the categories `emotional_abuse`, `physical_abuse`, `sexual_abuse`.

- [Social Care Institute for Excellence](https://www.scie.org.uk/safeguarding/adults/introduction/types-and-indicators-of-abuse/)
- [Washington State Department of Social and Health Services](https://www.dshs.wa.gov/altsa/home-and-community-services/types-and-signs-abuse)

The indicators for the category of interest are stored in a `.csv`-file called `[category]_indicators.csv` and need to have a column called `indicator`.
Any additional columns like `source` or `type` are not necessary for downstream tasks.

## Collect vocabulary words manually
When creating the indicator file, you can already collect words that are associated with the indicators.
Examples:

| Indicator                                                  | Word (POS-Tag) |
|------------------------------------------------------------|----------------|
| abuser leaving person unattended when they need assistance | abandon (VERB) |
| broken bones                                               | cracked (ADJ)  |

These should be stored in a separate file (not in `[category]_words.csv` as this will be overwritten by the GPT-4 suggestions).

## Let GPT-4 suggest vocabulary for the category
After creating the indicator file, the script [gpt4_suggestions.py](gpt4_suggestions.py) can be run as follows:
```
python -m ao3_tags.vocabulary.gpt4_suggestions [category] [openAI key]
```
Example:
```
python -m ao3_tags.vocabulary.gpt4_suggestions emotional_abuse [openAI key]
```
This will create a file called `[category]_words.csv` in [resources/categories](../../resources/categories). 

## Adding POS-Tags and manual cleaning of the vocabulary
The final step clean the GPT-4 suggestions and add your own manually collected words. 
- Remove any words that should not be part of the vocabulary
  - Words connected with a dash
  - Words with multiple meanings
  - ...
- Expand the list with missing words
- Clean the words
  - Ensure that words are stored as lemmas (e.g. no plurals of nouns)
- Add POS-Tags to the words
  - Add a column `pos_tag` and add `VERB`, `NOUN` or `ADJ` to the words
- Expand the list with other words with the same stem
  - `abandon` as a VERB: Add `abandoned` (ADJ), `abandonment` (NOUN), ...

Save the final `.csv`-file under its original name (`[category]_words.csv`) to be used in downstream tasks.