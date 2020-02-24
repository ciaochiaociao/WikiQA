#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import re
from typing import List

from ansi.colour import fg, bg
from stanfordnlp.protobuf import NERMention, Token, Document, Sentence

from utils import overlapped_range, apply_dict, print_multicolor_strs


def get_client(ip='http://140.109.19.191:9000', annotators="tokenize,ssplit,lemma,pos,ner"):
    return CoreNLPClient(endpoint=ip, annotators=annotators, start_server=False, properties='chinese')


def snp_get_ents_by_char_span_in_doc(span, snp_doc: Document):
    """
    get entity mentions of stanfordnlp document from character span (either contained or occupied by the span)

    :param tuple span: the character span that occupies or contains entities
    :param Document snp_doc: stanfordnlp Document class
    :return List[Entity]: list of entities
    """
    mentions = snp_doc.mentions
    all_tokens = [token for sent in snp_doc.sentence for token in sent.token]
    tok2char_map = snp_get_tok2char_map(all_tokens, snp_doc.text)

    mention_char_spans = []
    for mention in mentions:
        mention_token_span = snp_get_mention_token_span(mention)
        mention_char_span = toks2chars(mention_token_span, tok2char_map)
        mention_char_spans.append(mention_char_span)

    try:
        ixes, _ = overlapped_range(mention_char_spans, span)
        obtained_mentions = [mentions[ix] for ix in ixes]
        return obtained_mentions
        # print(f'NER not found in passage {span[0]} to {span[1]}')
    except TypeError:  # no overlapping
        return []


def snp_get_mention_token_span(snp_mention: NERMention):
    mention_token_span = snp_mention.tokenStartInSentenceInclusive, snp_mention.tokenEndInSentenceExclusive
    return mention_token_span


def snp_get_tok2char_map(snp_tokens: List[Token], text) -> List[int]:
    len_all_tokens = list(filter(bool, [len(token.originalText) for token in snp_tokens]))
    assert len(len_all_tokens) == len(snp_tokens)
    # build mapping
    s = 0
    tok_char_indexes = [s]
    for l in len_all_tokens:
        s += l
        tok_char_indexes.append(s)
    # solve the problem that tokenization does not include newlines
    for token in snp_tokens:
        tok_ix = token.tokenBeginIndex
        char_ix = tok_char_indexes[tok_ix]
        text_by_char_ix = text[char_ix: char_ix + len(token.originalText)]
        newlines = re.findall('[\n\s\t]', text_by_char_ix)
        if newlines:
            for ix in range(tok_ix, len(snp_tokens)):
                tok_char_indexes[ix] += len(newlines)

        # for debugging
        # tok_ix = token.tokenBeginIndex
        # char_ix = tok2char_map[tok_ix]
        # print(tok_ix, char_ix, token.originalText, passage_ie_data.text[char_ix: char_ix + len(token.originalText)])
    return tok_char_indexes


def toks2chars(token_indexes, tok2char_map):
    char_span = apply_dict(token_indexes, tok2char_map)
    return char_span


def snp_pprint(snp_sent: Sentence, mode: Union['color', 'bracket'] = 'color', **kwargs):
    tokens = snp_sent.token
    token_texts = [token.originalText for token in tokens]
    token_ners = [token.fineGrainedNER for token in tokens]

    color_table = {
        'PERSON': bg.red,
        'TITLE': fg.red,
        'NATIONALITY': lambda x: fg.red(bg.cyan(x)),
        'LOCATION': lambda x: fg.boldgreen(bg.cyan(x)),
        'GPE': fg.green,
        'CITY': bg.green,
        'STATE_OR_PROVINCE': bg.cyan,
        'COUNTRY': fg.cyan,
        'ORGANIZATION': bg.magenta,
        'FACILITY': lambda x: fg.blue(bg.magenta(x)),
        'MISC': fg.magenta,
        'DATE': bg.blue,
        'TIME': fg.blue,
        'NUMBER': fg.yellow,
        'PERCENT': lambda x: fg.blue(bg.yellow(x)),
        'MONEY': lambda x: fg.red(bg.yellow(x)),
        'ORDINAL': bg.yellow,
        'CAUSE_OF_DEATH': bg.white,
        'DEMONYM': bg.darkgray,
        'IDEOLOGY': lambda x: fg.magenta(bg.darkgray(x)),
        'RELIGION': lambda x: fg.yellow(bg.darkgray(x)),
        'CRIMINAL_CHARGE': lambda x: fg.red(bg.darkgray(x)),
        'O': fg.default
    }
    if mode == 'color':
        print_multicolor_strs(token_texts, token_ners, color_table, **kwargs)
    elif mode == 'bracket':
        pass  #TODO



