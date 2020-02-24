#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
from typing import Union, List

from stanfordnlp.protobuf import NERMention, Token

from stanfordnlp_utils import snp_get_ents_by_char_span_in_doc
from wikidata4fgc_v2 import get_dicts_from_keyword, clean_and_simplify_wd_items


def build_candidates_to_EL(name, question_ie_data, span, use_ner=True, split_dot=True):

    ent_link_cands = []

    # 1. parsed name from regex (default)
    ent_link_cands.append(name)

    # 2. add more candidates by post-processing (remove 'dot' in subject)
    # '.' or '·' in name, e.g., 馬可.波羅
    if split_dot:
        if '.' in name:
            ent_link_cands.append(''.join(name.split('.')))
        if '·' in name:
            ent_link_cands.append(''.join(name.split('·')))

    # 3. NER mentions containing/occupying the span
    if use_ner:
        mentions = list(snp_get_ents_by_char_span_in_doc(span, question_ie_data))
        if mentions:
            ent_link_cands.extend(mentions)

    return ent_link_cands


def entity_linking(ent_link_cands):

    # build queries
    ent_link_queries = build_queries(ent_link_cands)

    # query to Wikidata
    all_wd_items = []
    for ent_link_query in ent_link_queries:
        wd_items = get_dicts_from_keyword(ent_link_query)
        all_wd_items.extend(wd_items)

    # clean wikidata item and simplify wikidata item
    all_wd_items = clean_and_simplify_wd_items(all_wd_items)

    return all_wd_items


def build_queries(texts):
    shown_queries = set()
    ent_link_queries = []
    for ix, ent_link_cand in enumerate(ent_link_cands):
        ent_link_query = _get_text_from_token_entity_comp(ent_link_cand)
        if ent_link_query in shown_queries:  # skip duplicate
            continue
        shown_queries.add(ent_link_query)

        # ============================================================ ent_link_cand
        ent_link_queries.append(ent_link_query)
    return ent_link_queries


def _get_text_from_token_entity_comp(ent_link_cand: Union[List[Union[NERMention, Token]], str, NERMention]) -> str:
    """
    get text of `NERMention` and `Token`
    :type ent_link_cand: Union[List[Union[NERMention, Token]], str]
    :rtype: str
    """
    if isinstance(ent_link_cand, tuple):
        return ent_link_cand[0]
    if isinstance(ent_link_cand, NERMention):
        return ent_link_cand.entityMentionText
    if isinstance(ent_link_cand, str):
        return ent_link_cand