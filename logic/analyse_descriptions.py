from logic import constants as co
from logic import util as util
import spacy
from spacy.tokens import Span
from spacy.tokens import Token
import pandas as pd
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

output_dir = co.data_dir / co.custom_nlp_model_name
nlp = spacy.load(output_dir)


pair_with_re = [r"(enjoy(ed)?|pair(ed)?|\btry\b|\bserved?|combination|just the thing) .{,30}with",
                r"would make a great companion to",
                r"would be .{1,10}with",
                r"perfect( pairing)? for",
                r"(pairing|ideal|match) (for|to)",
                r"to (partner|perk up)",
                r"wants some .{,30}cooking to go with it",
                r"drink(ing)? .{0,10}with"]

verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def strip_extra_characters(tagged_words: co.TAGGED_WORDS) -> co.TAGGED_WORDS:
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
            if "FOOD_SEAFOOD" in categories:
                pairing_dict["FOOD_SEAFOOD"].append(pairing_text)
            elif "FOOD_MEAT_AND_POULTRY" in categories:
                pairing_dict["FOOD_MEAT_AND_POULTRY"].append(pairing_text)
            elif "FOOD_COUNTRY_CUISINE" in categories:
                pairing_dict["FOOD_COUNTRY_CUISINE"].append(pairing_text)
            else:
                # anything else but OTHER
                categories = [cat_ for cat_ in categories if cat_ != 'FOOD_OTHER']
                if categories:
                    pairing_dict[categories[0]].append(pairing_text)
    return pairing_dict


def is_false_positive(sentence: Span, noun_phrases: List[Span]) -> bool:
    """E.g. 'An easy drinking wine, with pleasantly tart cherry flavors.'"""
    is_fp = False
    food_ents = [ent.label_ for noun_phrase in noun_phrases for ent in noun_phrase.ents]
    if all([True if ent == 'FOOD_FRUITS' else False for ent in food_ents]) and\
            re.search(r"flavor|aroma|tannin", sentence.text, re.I):
        is_fp = True
    return is_fp


def filter_noun_phrases(noun_phrases: List[Span], start_ind: int) -> List[Span]:
    """Only leave phrases with FOOD entities, located after pairing regex."""
    filter_phrases = []
    for noun_phrase in noun_phrases:
        if noun_phrase.start_char < start_ind:
            continue
        for ent in noun_phrase.ents:
            if ent.label_ in co.food_labels:
                filter_phrases.append(noun_phrase)
                break
    return filter_phrases


def _find_index_in_list_of_noun_chunks(list_: List[Span], search_term: str):
    for i, item in enumerate(list_):
        if re.search(search_term, item.text, re.I):
            return i
    return -1


def find_head_tokens(phrase: Span) -> List[Token]:
    current_token = phrase.root
    head_tokens = []
    while True:
        if current_token.dep_ == 'ROOT':
            break
        head_tokens.append(current_token)
        next_index = current_token.head.i
        current_token = current_token.doc[next_index]
    return head_tokens


def _try_combine_phrases(phrase1: Span, phrase2: Span) -> Optional[Span]:
    phrase2_head_tokens = find_head_tokens(phrase2)
    current_token = phrase1.root
    i = 0
    reached_common_base = False
    while True:
        if i == 1 and current_token.tag_ not in ['IN']:
            break
        elif current_token in phrase2_head_tokens:
            reached_common_base = True
            break
        elif current_token.dep_ == 'ROOT' or current_token.tag_ in verbs:
            break
        next_index = current_token.head.i
        current_token = current_token.doc[next_index]
        i += 1
    if reached_common_base and i <= 2:
        combined_phrase = phrase1.doc[phrase1.start:phrase2.end]
    else:
        combined_phrase = None
    return combined_phrase


def find_potential_extension_index(noun_phrases: List[Span]) -> Union[int, List[int]]:
    """Find phrase the root of which has a 'with' head."""
    index_list = []
    for i, phrase in enumerate(noun_phrases):
        root = phrase.root
        if root.head.text in ['with', 'in']:
            index_list.append(i)
    if not index_list:
        return -1
    else:
        return index_list


def try_combine_phrases(noun_phrases: List[Span]) -> List[Span]:
    """Try combining phrases like 'roast duck' and 'fruit sauce' if they are connected with 'with'
    and are not too far from common base."""
    potential_ext_ind = find_potential_extension_index(noun_phrases)
    if potential_ext_ind == -1:
        return noun_phrases
    for ind in potential_ext_ind:
        if ind > 0:
            extension_phrase = noun_phrases[ind]
            food_phrase = noun_phrases[ind - 1]
            combined_phrase = _try_combine_phrases(food_phrase, extension_phrase)
            if combined_phrase:
                noun_phrases.remove(food_phrase)
                noun_phrases.remove(extension_phrase)
                noun_phrases.append(combined_phrase)
    return noun_phrases


def maybe_add_to_noun_phrases_from_food_ents(food_ents: List[Span], noun_phrases: List[Span], pairing_start: int) ->\
        List[Span]:
    """Add food entities not captured by noun chunks"""
    noun_phrases_str = ' '.join([n.text for n in noun_phrases])
    food_ents_not_in_noun_phrases = [food_ent for food_ent in food_ents
                                     if not re.search(food_ent.text, noun_phrases_str, re.I)
                                     and food_ent.start_char >= pairing_start]

    noun_phrases.extend(food_ents_not_in_noun_phrases)
    noun_phrases = sorted(noun_phrases, key=lambda x: x.start)
    return noun_phrases


def extract_food_pairings(sent_nlp: Span, pairing_start: int) -> dict:
    # [(t.text, t.tag_, t.head.text) for t in sent_nlp]
    noun_phrases = [chunk for chunk in sent_nlp.noun_chunks]
    noun_phrases = filter_noun_phrases(noun_phrases, pairing_start)
    food_spans = [ent for ent in sent_nlp.ents if ent.label_.startswith('FOOD_')]
    noun_phrases = maybe_add_to_noun_phrases_from_food_ents(food_spans, noun_phrases, pairing_start)
    pairings = []
    if is_false_positive(sent_nlp, noun_phrases):
        return {}
    if len(noun_phrases) > 1:
        noun_phrases = try_combine_phrases(noun_phrases)
    for noun_phrase in noun_phrases:
        text = noun_phrase.text
        entities = []
        for ent in noun_phrase.ents:
            if ent.label_ in co.food_labels:
                entities.append(ent.label_)
        pairings.append((text, entities))
    pairings_dict = organise_pairings_tuple_into_dict(pairings)
    return pairings_dict


def find_pairing_food(df: pd.DataFrame):
    data = deepcopy(df)
    for i, row in data[0:5000].iterrows():
        text = row[co.description]
        text_nlp = nlp(text)

        for sent_nlp in text_nlp.sents:
            sent = sent_nlp.text
        # for sent in sentences:
            if match := re.search('|'.join(pair_with_re), sent, re.I):
                pairing_dict = extract_food_pairings(sent_nlp, sent_nlp.start_char + match.end())
                if pairing_dict:
                    print(sent)
                    print(pairing_dict)
                else:
                    print(f"Non pairing sentence: {sent}")
                print('\n')
                for category, pairing_text in pairing_dict.items():
                    if category not in data.columns:
                        data[category] = pd.NaT
                    data.at[i, category] = '. '.join(pairing_text)
    return data


def analyse_descriptions(data: pd.DataFrame):
    my_food_dict = util.load_my_food()
    pairing_df = find_pairing_food(data)


if __name__ == "__main__":
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    analyse_descriptions(data)
