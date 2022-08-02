import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt

import constants as co


def load_my_food() -> Dict[str, List[str]]:
    with open(co.data_dir / 'food_list.txt', 'r') as f:
        my_food_list = f.read()

    categories = re.split(r"(?=#.*?#)", my_food_list)
    categories = [cat for cat in categories if cat]
    my_food_dict = defaultdict()
    for cat in categories:
        food_list = cat.split('\n')
        title = re.search(r"#(.*?)#", food_list[0])[1].lower()
        my_food_dict[title] = [fr"{food_item.lower().strip()}" for food_item in food_list[1:] if food_item]
    my_food_dict['other'].extend(co.weird_flavors)
    return my_food_dict


def check_if_keyword_present(string: str, list_of_keywords: Optional[List[str]]) -> bool:
    if not list_of_keywords:
        return False
    return any([True if re.search(regex, string, re.I) else False for regex in list_of_keywords])


def return_first_n_dict_items(dict_: dict, n: int) -> dict:
    return dict(itertools.islice(dict_.items(), n))


def format_text(text: str, max_words_on_line: int = 5) -> str:
    words = text.split(' ')
    new_str = ''
    i = 0
    for word in words:
        if '\n' in word:
            i = 0
        elif i > max_words_on_line - 1:
            i = 0
            new_str += '\n'
        new_str += ' ' + word
        i += 1
    return new_str.strip()


def display_text_box(text: str, axis=[0, 10, 0, 10], center_v: int = 5, center_h: int = 5, fontsize: int = 20) -> None:
    fig = plt.figure()
    plt.axis(axis)
    plt.text(center_v, center_h, text, ha='center', va='center', wrap=True, fontsize=fontsize)
    plt.axis('off')
    plt.grid(visible=None)
    plt.show()
