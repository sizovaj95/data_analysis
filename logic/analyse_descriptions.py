import constants as co
import util as util
import spacy
import nltk
import pandas as pd
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")


pair_with_re = [r"(enjoy(ed)?|pair(ed)?|\btry\b|\bserved?|combination|just the thing) .{,30}with",
                r"would make a great companion to",
                r"would be .{1,10}with",
                r"perfect( pairing)? for",
                r"(pairing|ideal|match) (for|to)",
                r"to (partner|perk up)",
                r"wants some .{,30}cooking to go with it",
                r"drink(ing)? .{0,10}with"]


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


def split_potential_pairings_into_parts_by_tags(tagged_words: co.TAGGED_WORDS) -> List[co.TAGGED_WORDS]:
    parts = []
    for word, tag in tagged_words:
        if tag == 'CC' or tag == ',':
            before_tag, after_tag = util.split_by_pos_tag(tagged_words, tag)
            parts.extend(split_potential_pairings_into_parts_by_tags(before_tag))
            parts.extend(split_potential_pairings_into_parts_by_tags(after_tag))
            break
    else:
        return [tagged_words]
    return parts


def split_and_strip_food_phrase(tagged_words: co.TAGGED_WORDS, custom_tags: List[str]):
    """E.g. split 'rich meats or mushrooms' into 'rich meats', 'mushrooms'"""

    pairing_parts = split_potential_pairings_into_parts_by_tags(tagged_words)
    pairings = []

    for i, part in enumerate(pairing_parts):
        if util.check_if_pos_tag_in_list(part, custom_tags):
            part = strip_extra_characters(part)
            pairings.append(part)
        else:
            part = _maybe_remove_part_of_pairing(part)
            if part != [('', '')]:
                try:
                    part += pairing_parts[i+1]
                    pairing_parts[i+1] = part
                except IndexError:
                    continue
    return pairings


def _maybe_remove_part_of_pairing(tagged_words: co.TAGGED_WORDS) -> co.TAGGED_WORDS:
    if len(tagged_words) == 1 and tagged_words[0][1] in ['JJ', 'NN', 'VBN']:
        return tagged_words
    else:
        return [('', '')]


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


def extract_pairing_food(tagged_words: co.TAGGED_WORDS, custom_tags: List[str]) -> dict:
    pairing_dict = {}
    i = 1
    for word, tag in tagged_words[::-1]:
        if tag in custom_tags:
            pairing_tup = [tup for tup in tagged_words[:-i+1]]
            pairings = split_and_strip_food_phrase(pairing_tup, custom_tags)
            pairings = organise_pairings_into_phrases_and_food_tags(pairings, custom_tags)
            pairing_dict = organise_pairings_tuple_into_dict(pairings)
            print(pairings)
            break
        i += 1
    return pairing_dict


def organise_pairings_into_phrases_and_food_tags(pairings: List[co.TAGGED_WORDS], custom_tags: List[str]) -> \
        List[Tuple[str, List[str]]]:
    pairings_with_pos = []
    for pairing in pairings:
        pairings_with_pos.append((' '.join(util.get_words(pairing)),
                                 util.get_only_custom_tags(pairing, custom_tags)))
    pairings_with_pos = [(phrase.strip(), pos) for phrase, pos in pairings_with_pos if pos]
    return pairings_with_pos


def get_tagged_words(words: List[str], regex_tagger: nltk.tag.RegexpTagger) -> co.TAGGED_WORDS:
    words = [w.lower() for w in words]
    tagged_words_default = nltk.pos_tag(words)
    tagged_words_regex = regex_tagger.tag(words)
    tagged_words = combine_results_of_two_taggers(tagged_words_default, tagged_words_regex)
    return tagged_words


def is_false_positive(match: re.Match, sentence: str, tagged_words_up_to: co.TAGGED_WORDS,
                      tagged_words_after: co.TAGGED_WORDS, custom_tags: List[str]) -> bool:
    tagged_words = tagged_words_up_to + tagged_words_after
    is_fp = False
    if paired_prepend := util.get_n_words_before_or_after_keyword(tagged_words, 'paired', 2, before_keyword=True):
        pos_tags = util.get_pos(paired_prepend)
        if not re.search(r"^(TO|MD) VB$", ' '.join(pos_tags)):
            is_fp = True
    if 'drink' in match[0]:
        food_tags = util.get_only_custom_tags(tagged_words, custom_tags)
        if len(food_tags) == 1 and food_tags[0] == "FRUITS" and re.search(r"flavor|aroma", sentence, re.I):
            is_fp = True
        elif before_drink := util.get_n_words_before_or_after_keyword(tagged_words, 'with', 1, before_keyword=True):
            if 'drink' not in before_drink[0][0] and before_drink[0][1] == 'VBN':
                is_fp = True
    return is_fp


def explore_spacy(sent_nlp):
    noun_phrases = [chunk for chunk in sent_nlp.noun_chunks]
    words = [(token.text, token.pos_, token.head.text, token.dep_) for token in sent_nlp]
    words


def find_pairing_food(df: pd.DataFrame, regex_tagger: nltk.tag.RegexpTagger, custom_tags: List[str]):
    data = deepcopy(df)
    for i, row in data[0:5000].iterrows():
        text = row[co.description]
        text_nlp = nlp(text)
        sentences = nltk.tokenize.sent_tokenize(text)

        for sent_nlp in text_nlp.sents:
            sent = sent_nlp.text
        # for sent in sentences:
            if match := re.search('|'.join(pair_with_re), sent, re.I):
                explore_spacy(sent_nlp)
                after_pair = sent[match.end():]
                before_pair = sent[:match.end()]
                after_words = nltk.tokenize.word_tokenize(after_pair)
                before_words = nltk.tokenize.word_tokenize(before_pair)

                tagged_words_after = get_tagged_words(after_words, regex_tagger)
                tagged_words_before = get_tagged_words(before_words, regex_tagger)
                # if is_false_positive(match, tagged_words_before + tagged_words_after):
                #     print(f"False positive: {sent}")
                #     continue

                pairing_dict = extract_pairing_food(tagged_words_after, custom_tags)
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
    my_food_dict = util.load_my_food()
    regex_tagger = define_custom_regex_tagger(my_food_dict)
    custom_tags = [t.upper() for t in my_food_dict.keys()]
    pairing_df = find_pairing_food(data, regex_tagger, custom_tags)
