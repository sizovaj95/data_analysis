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
import time
stop_words = set(stopwords.words("english"))

output_dir = co.data_dir / co.custom_nlp_model_name
nlp = spacy.load(output_dir)

my_food_dict = None
wine_adjectives = None


pair_with_re = [r"(enjoy(ed)?|pair(ed|ing)?|\btry\b|\bserved?|combination|just the thing) .{,30}with",
                r"would make a great companion to",
                r"would be .{1,10}with",
                r"perfect( pairing)? for",
                r"(pairing|ideal|match) (for|to)",
                r"to (partner|perk up)",
                r"wants some .{,30}cooking to go with it",
                r"drink(ing)? .{0,10}with"]

flavor_re = r"aroma|character|flavor|\bhints?\b|\btone\b|accents?\b|scents?\b"

verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def maybe_load_my_food_dict() -> Dict[str, List[str]]:
    global my_food_dict
    if my_food_dict is None:
        my_food_dict = util.load_my_food()
    return my_food_dict


def maybe_load_wine_adjectives() -> List[str]:
    global wine_adjectives
    if not wine_adjectives:
        with open(co.data_dir / 'words_to_describe_wine.txt', 'r') as f:
            wine_adj_str = f.read()
            wine_adjectives = [adj.lower() for adj in re.split(r"\n", wine_adj_str) if adj]
    return wine_adjectives


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
    if all([True if ent == 'FOOD_FRUITS' else False for ent in food_ents]):
        if re.search(r"flavor|aroma|tannin", sentence.text, re.I) or \
                (len(food_ents) == 1 and re.search('fruit', noun_phrases[0].text, re.I)):
            is_fp = True
    return is_fp


def filter_noun_phrases(noun_phrases: List[Span], start_ind: int, filter_list: Optional[List[str]] = None) ->\
        List[Span]:
    """Only leave phrases with FOOD entities or from 'filter_list', located after pairing regex."""
    filter_phrases = []
    for noun_phrase in noun_phrases:
        if noun_phrase.start_char < start_ind:
            continue
        if util.check_if_keyword_present(noun_phrase.text, filter_list):
            filter_phrases.append(noun_phrase)
            continue
        for ent in noun_phrase.ents:
            if ent.label_.startswith('FOOD_'):
                filter_phrases.append(noun_phrase)
                break
    return filter_phrases


def find_head_tokens(phrase: Span) -> List[Token]:
    """Find tokens along the dependency tree until reach the root of the sentence or after too many steps."""
    current_token = phrase.root
    head_tokens = []
    i = 0
    while i <= 10:
        if current_token.dep_ == 'ROOT':
            break
        head_tokens.append(current_token)
        next_index = current_token.head.i
        current_token = current_token.doc[next_index]
        i += 1
    return head_tokens


def _try_combine_phrases(phrase1: Span, phrase2: Span) -> Optional[Span]:
    """Combine phrases if they are connected.
    E.g. 'risotto' and 'grated cheese' into 'risotto with grated cheese'"""
    phrase2_head_tokens = find_head_tokens(phrase2)
    current_token = phrase1.root
    i = 0
    reached_common_base = False
    while i <= 10:
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
            try:
                extension_phrase = noun_phrases[ind]
            except IndexError:
                break
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


def extract_food_pairings(sent_nlp: Span, pairing_start: int) -> Dict[str, List[str]]:
    # [(t.text, t.tag_, t.head.text) for t in sent_nlp]
    noun_phrases = [chunk for chunk in sent_nlp.noun_chunks]
    noun_phrases = filter_noun_phrases(noun_phrases, pairing_start)
    food_spans = [ent for ent in sent_nlp.ents if ent.label_.startswith('FOOD_')]
    noun_phrases = maybe_add_to_noun_phrases_from_food_ents(food_spans, noun_phrases, pairing_start)
    if is_false_positive(sent_nlp, noun_phrases):
        return {}
    if len(noun_phrases) > 1:
        noun_phrases = try_combine_phrases(noun_phrases)
    pairings = organise_noun_phrases_into_food_categories(noun_phrases)
    pairings_dict = organise_pairings_tuple_into_dict(pairings)
    return pairings_dict


def organise_noun_phrases_into_food_categories(noun_phrases: List[Span]) -> List[Tuple[str, List[str]]]:
    """List of noun chinks into tuple of text and respective food category."""
    food_list = []
    for noun_phrase in noun_phrases:
        text = noun_phrase.text
        entities = []
        for ent in noun_phrase.ents:
            if ent.label_ in co.food_labels:
                entities.append(ent.label_)
        food_list.append((text, list(set(entities))))
    return food_list


def use_more_precise_food_categories(list_of_food_tuples: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """If food is of category OTHER, see if it can be given more precise category"""
    for i, (food_text, food_ent) in enumerate(list_of_food_tuples):
        if not food_ent:
            food_ent = ['FOOD_OTHER']
        for j, ent in enumerate(food_ent):
            if not ent.endswith('OTHER'):
                continue
            food_cat = identify_food_sub_category(food_text)
            if food_cat:
                food_ent[j] = food_cat
        list_of_food_tuples[i] = (food_text, food_ent)

    return list_of_food_tuples


def identify_food_sub_category(food_text: str) -> str:
    """Check food text against food examples in food dict used for training.
    A lot of categories were combined into 'OTHER', see if can be given subcategory, like 'SWEET'"""
    food_dict = maybe_load_my_food_dict()
    for food_cat, food_list in food_dict.items():
        if util.check_if_keyword_present(food_text, food_list):
            food_cat = f"FOOD_{food_cat}"
            food_cat = re.subn(r"\s", '', food_cat)[0].upper()
            return food_cat
    return ''


def extract_flavors(sent_nlp: Span) -> Dict[str, List[str]]:
    noun_phrases = [chunk for chunk in sent_nlp.noun_chunks]
    noun_phrases = filter_noun_phrases(noun_phrases, 0, co.weird_flavors)
    food_spans = [ent for ent in sent_nlp.ents if ent.label_.startswith('FOOD_')]
    noun_phrases = maybe_add_to_noun_phrases_from_food_ents(food_spans, noun_phrases, 0)
    flavors = organise_noun_phrases_into_food_categories(noun_phrases)
    flavors = use_more_precise_food_categories(flavors)
    flavors_dict = organise_pairings_tuple_into_dict(flavors)

    return flavors_dict


def add_food_info_to_df(df: pd.DataFrame):
    data = deepcopy(df)
    pairings_df = pd.DataFrame(index=list(data.index))
    flavor_df = pd.DataFrame(index=list(data.index))
    adjectives_df = pd.DataFrame(index=list(data.index))
    wine_adj = maybe_load_wine_adjectives()
    start = time.time()
    for i, row in data.iterrows():

        text = row[co.description]
        text_nlp = nlp(text)

        for sent_nlp in text_nlp.sents:
            sent = sent_nlp.text
            if match := re.search('|'.join(pair_with_re), sent, re.I):
                pairing_dict = extract_food_pairings(sent_nlp, sent_nlp.start_char + match.end())
                pairings_df = populate_df_with_dict(pairings_df, pairing_dict, i)
            if re.search(flavor_re, sent, re.I):
                flavor_dict = extract_flavors(sent_nlp)
                flavor_df = populate_df_with_dict(flavor_df, flavor_dict, i)
            if adjectives := re.findall(r'\b|\b'.join(wine_adj), sent, re.I):
                adj_dict = {"WINE_ADJECTIVES": adjectives}
                adjectives_df = populate_df_with_dict(adjectives_df, adj_dict, i)
        if i % 1000 == 0:
            print(f"Analysed row {i} / {len(data)}")
            print("Time taken for last 1000 rows: {:.2f} s".format(time.time() - start))
            print('\n')
            start = time.time()
    pairings_df.to_csv(co.data_dir / "food_pairings.csv", index=False)
    flavor_df.to_csv(co.data_dir / "food_flavors.csv", index=False)
    adjectives_df.to_csv(co.data_dir / "wine_description_words.csv", index=False)
    return data


def populate_df_with_dict(df: pd.DataFrame, dict_: dict, index: int) -> pd.DataFrame:
    for category, pairing_text in dict_.items():
        if category not in df.columns:
            df[category] = pd.NaT
        df.at[index, category] = '. '.join(pairing_text)
    return df


def analyse_descriptions(data: pd.DataFrame):
    add_food_info_to_df(data)


if __name__ == "__main__":
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    analyse_descriptions(data)
