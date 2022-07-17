from nltk.corpus import words
from typing import List, Dict
from nltk.corpus import gutenberg
import nltk
import json


def decompose(word: str, group_length: int) -> List[str]:
    """
    Decomposes a word into groups of letters.
    """
    return [word[i:i+group_length] for i in range(0, len(word) - group_length + 1)]


def extract_groups(words_list: List[str], filter_size=1000) -> Dict[int, List[str]]:
    """
    Extracts groups of words from a list of words.
    """
    group_size: int = 2
    all_groups: Dict[int, List[str]] = dict()

    while True:
        groups: Dict[str, int] = dict()
        for word in words.words():
            if len(word) > group_size:
                for group in decompose(word, group_size):
                    if group in groups.keys():
                        groups[group] += 1
                    else:
                        groups[group] = 1

        groups = {k: v for k, v in groups.items() if v > 10}
        group_filtered = dict(sorted(groups.items(), key=lambda item: item[1], reverse=True)[:filter_size])
        if len(group_filtered) <= 0:
            break

        all_groups[group_size] = group_filtered
        group_size += 1

    return all_groups


if __name__ == '__main__':

    # Extract groups from dictionary
    print(f"Parsing {len(words.words())} words")
    groups: Dict[int, List[str]] = extract_groups(words.words())
    with open(f'all_groups_english_words_1000.json', 'w') as fp:
        json.dump(groups, fp, indent=4)

    print("Saved all_groups_english_words_1000.json")

    # Extract groups from corpus
    corpus: List[str] = []
    for fileid in gutenberg.fileids():
        corpus += list(gutenberg.words(fileid))

    print(f"Parsing {len(corpus)} words")
    groups: Dict[int, List[str]] = extract_groups(corpus)
    with open(f'all_groups_corpus_1000.json', 'w') as fp:
        json.dump(groups, fp, indent=4)

    print("Saved all_groups_english_words_1000.json")
