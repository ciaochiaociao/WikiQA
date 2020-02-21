#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import re


def get_ent_from_tok_b_e(span, mentions, passage_data):
    for mention in mentions:
        # def tok_ix_2_char_ix

        all_tokens = [token for sent in passage_data.sentence for token in sent.token]
        len_all_tokens = list(filter(bool, [len(token.originalText) for token in all_tokens]))

        assert len(len_all_tokens) == len(all_tokens)
        s = 0
        tok_char_indexes = [s]
        for l in len_all_tokens:
            s += l
            tok_char_indexes.append(s)

        # solve the problem that tokenization does not include newlines
        for token in all_tokens:
            tok_ix = token.tokenBeginIndex
            char_ix = tok_char_indexes[tok_ix]
            text_by_char_ix = passage_data.text[char_ix: char_ix + len(token.originalText)]
            newlines = re.findall('[\n\s\t]', text_by_char_ix)
            if newlines:
                for ix in range(tok_ix, len(all_tokens)):
                    tok_char_indexes[ix] += len(newlines)

            # for debugging
            # tok_ix = token.tokenBeginIndex
            # char_ix = tok_char_indexes[tok_ix]
            # print(tok_ix, char_ix, token.originalText, passage_data.text[char_ix: char_ix + len(token.originalText)])
        mention_slice = (tok_char_indexes[mention.tokenStartInSentenceInclusive], tok_char_indexes[mention.tokenEndInSentenceExclusive])
        # print(mention_slice, span)
        # print(passage_data.text[slice(*mention_slice)], passage_data.text[slice(*span)])
        if set(range(*mention_slice)).intersection(range(*span)):
            yield mention
    # print(f'NER not found in passage {span[0]} to {span[1]}')