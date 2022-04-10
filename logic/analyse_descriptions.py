import nltk
import pandas as pd
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

import constants as co


pair_with_re = [r"(enjoy(ed)?|pair(ed?)|\btry\b|served?|drink|combination|just the thing) .{,30}with",
                r"would make a great companion to",
                r"would be .{1,10}with",
                r"perfect( pairing)? for",
                r"(pairing|ideal|match) (for|to)",
                r"to (partner|perk up)",
                r"wants some .{,30}cooking to go with it",
                r"with wide variety of (fare|food|dish)"]

TAGGED_WORDS = List[Tuple[str, str]]


class TaggedWordsUtil:
    def __init__(self, tagged_words: TAGGED_WORDS, custom_tags: List[str]):
        self.tagged_words = tagged_words
        self.custom_tags = custom_tags

    @property
    def words_only(self) -> List[str]:
        return self.get_words(self.tagged_words)

    @property
    def pos_only(self) -> List[str]:
        return self.get_pos(self.tagged_words)

    def get_only_custom_tags(self, tagged_words: TAGGED_WORDS):
        return [tag[1] for tag in tagged_words if tag[1] in self.custom_tags]

    @staticmethod
    def get_words(tagged_words: TAGGED_WORDS):
        return [tup[0] for tup in tagged_words]

    @staticmethod
    def get_pos(tagged_words: TAGGED_WORDS):
        return [tup[1] for tup in tagged_words]

    def check_if_pos_tag_in_list(self, tagged_words: TAGGED_WORDS, check_tag_list: List[str]) -> bool:
        for check_tag in check_tag_list:
            if check_tag in self.get_pos(tagged_words):
                return True
        return False

    @staticmethod
    def split_by_pos_tag(tagged_words: TAGGED_WORDS, splitting_tag: str) -> Tuple[TAGGED_WORDS, TAGGED_WORDS]:
        before_tag = []
        after_tag = []
        tag_found = False
        for word, tag in tagged_words:
            if tag == splitting_tag:
                tag_found = True
                continue
            if not tag_found:
                before_tag.append((word, tag))
            else:
                after_tag.append((word, tag))
        return before_tag, after_tag


def load_my_food() -> dict:
    with open(co.data_dir / 'food_list.txt', 'r') as f:
        my_food_list = f.read()

    categories = re.split(r"(?=#.*?#)", my_food_list)
    categories = [cat for cat in categories if cat]
    my_food_dict = defaultdict()
    for cat in categories:
        food_list = cat.split('\n')
        title = re.search(r"#(.*?)#", food_list[0])[1].lower()
        my_food_dict[title] = [fr"{food_item.lower().strip()}" for food_item in food_list[1:] if food_item]
    return my_food_dict


def define_custom_regex_tagger(food_dict: Dict[str, List[str]]) -> nltk.tag.RegexpTagger:
    patterns = []
    for food_type, food_list in food_dict.items():
        patterns_ = [(food_item, food_type.upper()) for food_item in food_list]
        patterns.extend(patterns_)
    regex_tagger = nltk.tag.RegexpTagger(patterns)
    return regex_tagger


def combine_results_of_two_taggers(tagged_words_main: List[tuple], tagged_words_add: List[tuple]) -> List[tuple]:
    try:
        assert len(tagged_words_main) == len(tagged_words_add)
    except AssertionError:
        raise AssertionError(f"Length of tagged_words_main {len(tagged_words_main)} "
                             f"is not equal to length of tagged_words_add {len(tagged_words_add)}")
    i = 0
    for tagged_main, tagged_add in zip(tagged_words_main, tagged_words_add):
        if tagged_add[1]:
            tagged_words_main[i] = tagged_add
        i += 1
    return tagged_words_main


def split_and_strip_food_phrase(tagged_words_util: TaggedWordsUtil) -> List[Tuple[str, List[str]]]:
    """E.g. split 'rich meats or mushrooms' into 'rick meats', 'mushrooms'"""
    tagged_words = tagged_words_util.tagged_words
    tagged_words = strip_extra_characters(tagged_words)
    pairing_as_str = ' '.join(tagged_words_util.get_words(tagged_words))
    pairings_with_pos = [(pairing_as_str, tagged_words_util.get_only_custom_tags(tagged_words))]
    for word, tag in tagged_words:
        if tag == "CC":
            before_tag, after_tag = tagged_words_util.split_by_pos_tag(tagged_words, 'CC')
            pairings_with_pos = \
                [(' '.join(tagged_words_util.get_words(before_tag)), tagged_words_util.get_only_custom_tags(before_tag)),
                 (' '.join(tagged_words_util.get_words(after_tag)), tagged_words_util.get_only_custom_tags(after_tag))]

        elif tag == ",":
            before_tag, after_tag = tagged_words_util.split_by_pos_tag(tagged_words, ',')
            if tagged_words_util.check_if_pos_tag_in_list(before_tag, tagged_words_util.custom_tags) and\
                    tagged_words_util.check_if_pos_tag_in_list(after_tag, tagged_words_util.custom_tags):
                pairings_with_pos = \
                    [(' '.join(tagged_words_util.get_words(before_tag)), tagged_words_util.get_only_custom_tags(before_tag)),
                     (' '.join(tagged_words_util.get_words(after_tag)), tagged_words_util.get_only_custom_tags(after_tag))]
            else:
                pairings_with_pos = \
                    [(' '.join(tagged_words_util.words_only),
                      tagged_words_util.get_only_custom_tags(tagged_words))]

    pairings_with_pos = [(phrase.strip(), pos) for phrase, pos in pairings_with_pos if pos]
    return pairings_with_pos


def strip_extra_characters(tagged_words: TAGGED_WORDS) -> TAGGED_WORDS:
    """E.g. 'a steak' -> 'steak'"""
    clean_tagged_words = deepcopy(tagged_words)
    for word, tag in tagged_words:
        if tag in ["DT", "IN", ","]:
            clean_tagged_words.remove((word, tag))
        else:
            break
    return clean_tagged_words


def organise_pairings_tuple_into_dict(pairing_tuples_list: List[Tuple[str, List[str]]]) -> Dict[str, list]:
    """From
    [('shellfish', ['SEAFOOD'])] to
    {'SEAFOOD': ['shellfish']}"""
    pairing_dict = defaultdict(list)
    for pairing_text, categories in pairing_tuples_list:
        if len(categories) == 1:
            pairing_dict[categories[0]].append(pairing_text)
        else:
            # try to pick main pairing (meat, seafood)
            if "SEAFOOD" in categories:
                pairing_dict["SEAFOOD"].append(pairing_text)
            elif "MEAT AND POULTRY" in categories:
                pairing_dict["MEAT AND POULTRY"].append(pairing_text)
            else:
                # anything else but OTHER
                categories = [cat_ for cat_ in categories if cat_ != 'OTHER']
                if categories:
                    pairing_dict[categories[0]].append(pairing_text)
    return pairing_dict


def extract_pairing_food(tagged_words_util: TaggedWordsUtil) -> dict:
    pairing_dict = {}
    i = 1
    for word, tag in tagged_words_util.tagged_words[::-1]:
        if tag in tagged_words_util.custom_tags:
            pairing_tup = [tup for tup in tagged_words_util.tagged_words[:-i+1]]
            tagged_words_util.tagged_words = pairing_tup
            pairings = split_and_strip_food_phrase(tagged_words_util)
            pairing_dict = organise_pairings_tuple_into_dict(pairings)
            print(pairings)
            break
        i += 1
    return pairing_dict


def get_tagged_words(words: List[str], regex_tagger: nltk.tag.RegexpTagger) -> TAGGED_WORDS:
    words = [w.lower() for w in words]
    tagged_words_default = nltk.pos_tag(words)
    tagged_words_regex = regex_tagger.tag(words)
    tagged_words = combine_results_of_two_taggers(tagged_words_default, tagged_words_regex)
    return tagged_words


def find_pairing_food(df: pd.DataFrame, regex_tagger: nltk.tag.RegexpTagger, custom_tags: List[str]):
    data = deepcopy(df)
    for i, row in data[1500:5000].iterrows():
        text = row[co.description]
        sentences = nltk.tokenize.sent_tokenize(text)
        for sent in sentences:
            if match := re.search('|'.join(pair_with_re), sent, re.I):
                if 'Asian' in sent:
                    print()
                after_pair = sent[match.end():]
                before_pair = sent[:match.end()]
                after_words = nltk.tokenize.word_tokenize(after_pair)
                before_words = nltk.tokenize.word_tokenize(before_pair)
                tagged_words_ = get_tagged_words(after_words, regex_tagger)
                tagged_words = TaggedWordsUtil(tagged_words_, custom_tags)

                pairing_dict = extract_pairing_food(tagged_words)
                if pairing_dict:
                    print(sent)
                else:
                    print(f"Non pairing sentence: {sent}")
                print('\n')
                for category, pairing_text in pairing_dict.items():
                    if category not in data.columns:
                        data[category] = pd.NaT
                    data.at[i, category] = '. '.join(pairing_text)
    return data


def analyse_descriptions(data: pd.DataFrame):
    my_food_dict = load_my_food()
    regex_tagger = define_custom_regex_tagger(my_food_dict)
    custom_tags = [t.upper() for t in my_food_dict.keys()]
    pairing_df = find_pairing_food(data, regex_tagger, custom_tags)
