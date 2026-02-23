from resiliparse.extract.html2text import extract_plain_text
from typing import Dict, Iterator, Sequence, Set


def tokenize_chapter(element: Dict, nlp, stopwords: Set[str], pos_tags: Sequence[str] = None) -> Iterator:
    pos_tags = pos_tags or ["VERB", "ADJ", "NOUN"]

    # Get the relevant parts from the element
    try:
        chap_content = element['fields']['chap_content']
        chapter_id = element['fields']['_id'][0]
        plain_text = extract_plain_text(chap_content[0])
    except:
        return None

    # Split the chapter content into words
    for word in nlp(plain_text):
        if (word.pos_ in pos_tags) and (word.text.lower() not in stopwords) and (not word.is_stop):
            yield {'chapter_id': chapter_id,
                   'word': word.lemma_.lower(),
                   'pos_tag': word.pos_,
                   'tf': 1
                   }