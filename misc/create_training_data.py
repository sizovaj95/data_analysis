import pandas as pd
import nltk
from typing import Tuple, Dict, List, Optional
import re
import pickle
import json

from logic import constants as co
from logic import util as util


TRAINING_DATA = []


def convert_sentence_into_training_item(sentence: str, food_dict: Dict[str, List[str]]) -> Optional[Tuple[str, dict]]:
    main_food_cat = ["meat and poultry", "seafood", "general dishes", "country cuisine"]
    entities = []
    for food_cat, food in food_dict.items():
        if food_cat not in main_food_cat:
            food_cat = 'other'
        food_cat = f"food_{food_cat}".upper()
        food_re = r"|".join(food)
        if matches := list(re.finditer(food_re, sentence, re.I)):
            for match in matches:
                entities.append((match.start(), match.end(), food_cat))
    if entities:
        entities = remove_overlapping_entities(entities)
        training_item = (sentence, {'entities': entities})
    else:
        training_item = None
    return training_item


def adjust_food_dict(food_dict: Dict[str, List[str]]):
    countries = food_dict["country cuisine"]
    new_values = []
    food_re = r"\s(?:food|dish|cuisine|fare)"
    for country in countries:
        new_values.append(country + food_re)
    food_dict["country cuisine"] = new_values
    return food_dict


def remove_overlapping_entities(entities_list: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Example:
    [(91, 102, 'FOOD_OTHER'), (97, 102, 'FOOD_OTHER')] -> [(91, 102, 'FOOD_OTHER')]"""
    entities_list = list(set(entities_list))
    sorted_by_start = sorted(entities_list, key=lambda x: x[0])
    for i, (start_, _, _) in enumerate(sorted_by_start):
        try:
            if start_ == sorted_by_start[i+1][0]:
                min_match = min([sorted_by_start[i], sorted_by_start[i+1]], key=lambda x: x[1])
                entities_list.remove(min_match)
        except IndexError:
            break
    sorted_by_end = sorted(entities_list, key=lambda x: x[1])
    for i, (_, end_, _) in enumerate(sorted_by_end):
        try:
            if end_ == sorted_by_end[i+1][1]:
                min_match = max([sorted_by_end[i], sorted_by_end[i+1]], key=lambda x: x[0])
                entities_list.remove(min_match)
        except IndexError:
            break
    return entities_list


def main():
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    food_dict = util.load_my_food()
    food_dict = adjust_food_dict(food_dict)
    for i, row in data.iterrows():
        description = row[co.description]
        sentences = nltk.tokenize.sent_tokenize(description)
        for sentence in sentences:
            training_example = convert_sentence_into_training_item(sentence, food_dict)
            if training_example:
                TRAINING_DATA.append(training_example)
    with open(co.data_dir / "ner_training_data.json", "w") as f:
        json.dump(TRAINING_DATA, f)


if __name__ == "__main__":
    with open(co.data_dir / "ner_training_data.json", 'r') as f:
        data = json.load(f)
        data
