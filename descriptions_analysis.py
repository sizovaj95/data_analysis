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


def main():
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    my_food_dict, my_food_list = load_my_food()
    regex_tagger = define_custom_regex_tagger(my_food_dict)
    for i, row in data[:1500].iterrows():
        text = row[data]
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
                # print(sent)
                # print(tagged_words)
                # print('\n')
                # print(sent)
                # print('\n')


if __name__ == "__main__":
    main()

