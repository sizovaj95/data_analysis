from typing import List, Tuple, Dict, Optional

import constants as co


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
