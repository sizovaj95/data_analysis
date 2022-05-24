import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from logic import constants as co


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
