#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import re
from typing import List, Union

from ansi.colour import fg, bg
from stanfordnlp.protobuf import NERMention, Token, Document, Sentence
from stanfordnlp.server import CoreNLPClient

from .utils import ranges_overlapped_w_range, apply_dict, get_multicolor_strs, overlapped


def get_client(ip='http://140.109.19.191:9000', annotators="tokenize,ssplit,lemma,pos,ner"):
    return CoreNLPClient(endpoint=ip, annotators=annotators, start_server=False, properties='chinese')


def snp_get_ents_by_overlapping_char_span_in_doc(span, snp_doc):
    all_tokens = [token for sent in snp_doc.sentence for token in sent.token]
    mentions = []
    for mention in snp_doc.mentions:
        _span = snp_get_char_span_of_ent(all_tokens, mention)
        assert snp_doc.text[slice(*_span)] == mention.entityMentionText
        if overlapped(span, _span):
            mentions.append(mention)
    return mentions


def snp_get_char_span_of_ent(all_tokens, mention):
    tokens = all_tokens[mention.tokenStartInSentenceInclusive:mention.tokenEndInSentenceExclusive]
    _span = tokens[0].beginChar, tokens[-1].endChar
    return _span


def snp_get_char_span_of_ent_in_sent(all_tokens, mention, doc: Document):
    _span = snp_get_char_span_of_ent(all_tokens, mention)
    sent_char_offset = doc.sentence[mention.sentenceIndex].characterOffsetBegin
    _span = _span[0] - sent_char_offset, _span[1] - sent_char_offset
    return _span


def snp_get_ner_tuples_from_sent(sent: Sentence, doc: Document):
    ner_tuples = [(*snp_get_char_span_of_ent_in_sent([t for s in doc.sentence for t in s.token], m, doc), m.ner) for m in
                  sent.mentions]
    return ner_tuples


def snp_get_ner_tuples_from_doc(doc: Document):
    ner_tuples = [(*snp_get_char_span_of_ent([t for s in doc.sentence for t in s.token], m), m.ner) for m in
                  doc.mentions]
    return ner_tuples


def snp_get_sent_text(sent, doc):
    return doc.text[sent.token[0].beginChar:sent.token[-1].endChar]


def snp_pprint_by_displacy(data, doc=None, **kwargs):
    from spacy import displacy

    if isinstance(data, Document):
        ner_tuples = snp_get_ner_tuples_from_doc(data)
        text = data.text
    elif isinstance(data, Sentence):
        ner_tuples = snp_get_ner_tuples_from_sent(data, doc)
        text = snp_get_sent_text(data, doc)
    else:
        raise ValueError
    ner_dicts = [{"start": start, "end": end, "label": ner} for start, end, ner in ner_tuples]

    ex = [{"text": text,
           "ents": ner_dicts,
           "title": None}]
    colors = {
        "ORG": "#7aecec",
        "ORGANIZATION": "#7aecec",
        "FACILITY": "#7aecec",

        "LOC": "#feca74",
        "LOCATION": "#feca74",

        "GPE": "#ff9561",
        "COUNTRY": "#ff9561",
        "STATE_OR_PROVINCE": "#ff9561",
        "CITY": "#ff9561",
        "DEMONYM": "#ff9561",

        "NATIONALITY": "#ffeb80",

        "PER": "#aa9cfc",
        "PERSON": "#aa9cfc",

        "TITLE": "#9cc9cc",

        "EVENT": "#ff8197",
        "CAUSE_OF_DEATH": "#ff8197",
        "CRIMINAL_CHARGE": "#ff8197",

        "RELIGION": "#f0d0ff",
        "IDEOLOGY": "#c887fb",
        "MISC": "#bfeeb7",

        "DATE": "#bfe1d9",
        "TIME": "#bfe1d9",

        "MONEY": "#e4e7d2",
        "QUANTITY": "#e4e7d2",
        "ORDINAL": "#e4e7d2",
        "CARDINAL": "#e4e7d2",
        "PERCENT": "#e4e7d2",
    }
    displacy.render(ex, style="ent", manual=True, jupyter=True, options={'colors': colors}, **kwargs)


def snp_get_ents_by_char_span_in_doc_old(span, snp_doc: Document):
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

    assert len(mention_char_spans) == len(mentions)
    for char_span, mention in zip(mention_char_spans, mentions):
        assert snp_doc.text[slice(*char_span)] == mention.entityMentionText, \
        ' '.join(['char_span: ', str(char_span[0]), str(char_span[1]), snp_doc.text[slice(*char_span)], 'mention:', mention.entityMentionText])

    try:
        ixes, _ = ranges_overlapped_w_range(mention_char_spans, span)
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
        'DYNASTY': lambda x: fg.boldgreen(bg.blue(x)),
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


def snp_print_legends():
    print(get_multicolor_strs(color_table.keys(), color_table.keys(), color_table, all_classes=color_table.keys()))


def snp_pstr(snp_sent: Sentence, mode: Union['color', 'bracket', 'custom'] = 'color'):
    tokens = snp_sent.token
    token_texts = [token.originalText for token in tokens]
    token_ners = [token.fineGrainedNER for token in tokens]

    if mode == 'color':
        return get_multicolor_strs(token_texts, token_ners, color_table, all_classes=color_table.keys())
    elif mode == 'custom':
        return get_multicolor_strs(token_texts, token_ners, all_classes=color_table.keys())
    elif mode == 'bracket':
        pass  #TODO



