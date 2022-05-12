import re
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
from collections import defaultdict

from logic import constants as co


def get_only_custom_tags(tagged_words: co.TAGGED_WORDS, custom_tags: List[str]) -> List[str]:
    return [tag[1] for tag in tagged_words if tag[1] in custom_tags]


def get_words(tagged_words: co.TAGGED_WORDS) -> List[str]:
    return [tup[0] for tup in tagged_words]


def get_pos(tagged_words: co.TAGGED_WORDS) -> List[str]:
    return [tup[1] for tup in tagged_words]


def check_if_pos_tag_in_list(tagged_words: co.TAGGED_WORDS, check_tag_list: List[str]) -> bool:
    for check_tag in check_tag_list:
        if check_tag in get_pos(tagged_words):
            return True
    return False


def split_by_pos_tag(tagged_words: co.TAGGED_WORDS, splitting_tag: str) -> Tuple[co.TAGGED_WORDS, co.TAGGED_WORDS]:
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


def get_n_words_before_or_after_keyword(tagged_words: co.TAGGED_WORDS, keyword: str, n: int, before_keyword: bool) ->\
        co.TAGGED_WORDS:
    tagged_words_ = deepcopy(tagged_words)
    if len(tagged_words_) <= n:
        return []
    if not re.search(keyword, ' '.join(get_words(tagged_words_)), re.I):
        return []
    selected_words = []
    if before_keyword:
        tagged_words_.reverse()
    for i, (word, pos) in enumerate(tagged_words_):
        if word == keyword:
            selected_words.extend(tagged_words_[i+1:i+n+1])
            break

    if before_keyword:
        selected_words.reverse()
    return selected_words


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
