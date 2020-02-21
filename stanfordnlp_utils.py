#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import re
from typing import List


def get_ent_from_stanford_by_char_span(span, mentions, stanfordnlp_data):

    all_tokens = [token for sent in stanfordnlp_data.sentence for token in sent.token]
    tok2char_map = get_tok2char_map(all_tokens, stanfordnlp_data.text)

    mention_char_spans = []
    for mention in mentions:
        mention_token_span = get_mention_token_span(mention)
        mention_char_span = toks2chars(mention_token_span, tok2char_map)
        mention_char_spans.append(mention_char_span)

    try:
        ix, _ = overlapped_range(mention_char_spans, span)
        obtained_mention = mentions[ix]
        return [obtained_mention]
        # print(f'NER not found in passage {span[0]} to {span[1]}')
    except TypeError:  # no overlapping
        return []


def overlapped_range(ranges, overlapper):
    for ix, _range in enumerate(ranges):
        if set(range(*_range)).intersection(range(*overlapper)):
            return ix, _range


def toks2chars(token_indexes, tok2char_map):
    char_span = (tok2char_map[token_indexes[0]],
                 tok2char_map[token_indexes[1]])
    return char_span


def get_mention_token_span(mention):
    mention_token_span = mention.tokenStartInSentenceInclusive, mention.tokenEndInSentenceExclusive
    return mention_token_span


def get_tok2char_map(tokens, text) -> List[int]:
    len_all_tokens = list(filter(bool, [len(token.originalText) for token in tokens]))
    assert len(len_all_tokens) == len(tokens)
    # build mapping
    s = 0
    tok_char_indexes = [s]
    for l in len_all_tokens:
        s += l
        tok_char_indexes.append(s)
    # solve the problem that tokenization does not include newlines
    for token in tokens:
        tok_ix = token.tokenBeginIndex
        char_ix = tok_char_indexes[tok_ix]
        text_by_char_ix = text[char_ix: char_ix + len(token.originalText)]
        newlines = re.findall('[\n\s\t]', text_by_char_ix)
        if newlines:
            for ix in range(tok_ix, len(tokens)):
                tok_char_indexes[ix] += len(newlines)

        # for debugging
        # tok_ix = token.tokenBeginIndex
        # char_ix = tok2char_map[tok_ix]
        # print(tok_ix, char_ix, token.originalText, passage_ie_data.text[char_ix: char_ix + len(token.originalText)])
    return tok_char_indexes