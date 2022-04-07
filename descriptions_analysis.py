from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
# nltk.download('omw-1.4')
# nltk.download("stopwords")
# nltk.download('words')
# nltk.download('brown')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud

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


def load_food_from_nltk():
    # TODO maybe don't need it
    # FOOD WORDS FROM NLTK
    # taken from https://stackoverflow.com/questions/19626737/where-can-i-find-a-text-list-or-library-that-contains-a-list-of-common-foods
    food = wn.synset('food.n.02')
    food_list_nltk = list(set([w for s in food.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    food_list_nltk = [re.subn("_", " ", i)[0].lower() for i in food_list_nltk]
    return food_list_nltk


def load_my_food() -> Tuple[dict, list]:
    with open(co.data_dir / 'food_list.txt', 'r') as f:
        my_food_list = f.read()

    categories = re.split(r"(?=#.*?#)", my_food_list)
    categories = [cat for cat in categories if cat]
    my_food_dict = defaultdict()
    for cat in categories:
        food_list = cat.split('\n')
        title = re.search(r"#(.*?)#", food_list[0])[1].lower()
        my_food_dict[title] = [fr"{food_item.lower().strip()}" for food_item in food_list[1:] if food_item]
    my_food_list = sum(my_food_dict.values(), [])
    my_food_list = [fi for fi in my_food_list if fi]
    return my_food_dict, my_food_list


def add_nltk_food_to_my_food(my_food_dict: Dict[str, List[str]], my_food_list: List[str]) -> Dict[str, List[str]]:
    # TODO maybe don't need it
    # food that is present in nltk list only
    food_list_nltk = load_food_from_nltk()
    nltk_food_as_str = ', '.join(food_list_nltk)
    for my_food_item in my_food_list:
        if re.search(r"(^|, ){}(, |$)".format(my_food_item), nltk_food_as_str):
            nltk_food_as_str = re.sub(my_food_item, '', nltk_food_as_str)
    nltk_food_only = nltk_food_as_str.split(', ')
    nltk_food_only = [fi.strip() for fi in nltk_food_only if len(fi) > 2]
    my_food_dict['nltk food'] = nltk_food_only
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


def split_and_strip_food_phrase(tagged_words: TAGGED_WORDS, custom_tags: List[str]):
    """
    rich meats or mushrooms
    meat or cheese
    hearty pasta or rich meats
    grilled beef or lamb
    pork ribs or flank steak
    casual chicken or fish
    chocolate or aged cheese
    roast chicken in herbes de Provence or ratatouille

    roasted meats and savory pies
    shellfish and salads
    halibut and shellfish
    sunny days and shellfish

    simple{NN} , white(JJ) fish(SEAFOOD)
    lamb(MEAT) , fresh(JJ) fish(SEAFOOD)

    make look nicer
    , like sautéed fish with a classic French sauce
    a platter of grilled meats
    that shellfish
    a steak
    simple , white fish

    leave as is
    elaborate roasts and stews
    """
    pairing_as_str = ' '.join([tup[0] for tup in tagged_words])
    pos_as_str = ' '.join([tup[1] for tup in tagged_words])
    pairings = [pairing_as_str]
    descriptive_pos = ["JJ", "VBN", "NN"]
    for word, tag in tagged_words:
        if tag == "CC":
            before_tag, after_tag = _split_by_pos_tag(tagged_words, 'CC')
            pairings = [' '.join(_get_words_list(before_tag)), ' '.join(_get_words_list(after_tag))]
            # if word.lower() == 'or':
            #     pairings = pairing_as_str.split('or')
            # elif word.lower() == "and":
            #     if len(tagged_words) == 3 and (tagged_words[0][1] in custom_tags and tagged_words[2][1] in custom_tags):
            #         pairings = pairing_as_str.split('and')
        elif tag == ",":
            before_tag, after_tag = _split_by_pos_tag(tagged_words, ',')
            if _check_if_pos_tag_in_list(before_tag, custom_tags) and _check_if_pos_tag_in_list(after_tag, custom_tags):
                pairings = [' '.join(_get_words_list(before_tag)), ' '.join(_get_words_list(after_tag))]
            else:
                pairings = [' '.join(_get_words_list(tagged_words))]

    pairings = [p.strip() for p in pairings]
    return pairings


def _get_words_list(tagged_words: TAGGED_WORDS) -> List[str]:
    return [tup[0] for tup in tagged_words]


def _get_pos_list(tagged_words: TAGGED_WORDS) -> List[str]:
    return [tup[1] for tup in tagged_words]


def _check_if_pos_tag_in_list(tagged_words: TAGGED_WORDS, check_tag_list: List[str]) -> bool:
    pos_list = _get_pos_list(tagged_words)
    for check_tag in check_tag_list:
        if check_tag in pos_list:
            return True
    return False


def _split_by_pos_tag(tagged_words: TAGGED_WORDS, splitting_tag: str) -> Tuple[TAGGED_WORDS, TAGGED_WORDS]:
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


def extract_pairing_food(tagged_words: TAGGED_WORDS, custom_tags: List[str]) -> list:
    descriptive_pos = ["JJ", "VBN", "NN"]
    pairings = []
    i = 1
    for word, tag in tagged_words[::-1]:
        if tag in custom_tags:
            pairing_tup = [tup for tup in tagged_words[:-i+1]]
            pairings = split_and_strip_food_phrase(pairing_tup, custom_tags)
            pairing = [tup[0] for tup in tagged_words[:-i+1]]
            food_category = [tup[1] for tup in tagged_words[:-i+1] if tup[1] in custom_tags]
            pairing = ' '.join(pairing)
            print(pairings)
            break
        i += 1
    return pairings


def main():
    # TODO make it all a class (anything pos tagging related)
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    my_food_dict, my_food_list = load_my_food()
    regex_tagger = define_custom_regex_tagger(my_food_dict)
    custom_tags = [t.upper() for t in my_food_dict.keys()]
    for i, row in data[:1500].iterrows():
        text = row[co.description]
        sentences = nltk.tokenize.sent_tokenize(text)
        for sent in sentences:
            if match := re.search('|'.join(pair_with_re), sent, re.I):
                after_pair = sent[match.end():]
                before_pair = sent[:match.end()]
                after_words = nltk.tokenize.word_tokenize(after_pair)
                before_words = nltk.tokenize.word_tokenize(before_pair)
                tagged_after_words_default = nltk.pos_tag(after_words)
                tagged_after_words_regex = regex_tagger.tag(after_words)
                tagged_words = combine_results_of_two_taggers(tagged_after_words_default, tagged_after_words_regex)
                pairing = extract_pairing_food(tagged_words, custom_tags)
                if pairing:
                    print(sent)
                    print('\n')
                else:
                    print(f"No pairing sentence: {sent}")
                # print(sent)
                # print(tagged_words)
                # print('\n')
                # print(sent)
                # print('\n')


if __name__ == "__main__":
    main()

