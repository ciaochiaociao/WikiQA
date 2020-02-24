#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import re
from typing import List

from config import cc
from wikidata4fgc_v2 import get_all_aliases_from_dict, get_all_aliases_from_id


def traverse_by_attr_name(wd_item, attr):
    # match predicates
    # new version:
    def _get_value_from_attr_and_wd_item(wd_item, attr):
        if attr == '名字':
            aliases = get_all_aliases_from_dict(wd_item)
            yield (('名字'),), aliases[0] + aliases[1]
        for rel, datavalues in wd_item['claims'].items():
            labels, _ = rel
            for label in labels:
                if label == attr:
                    # print('matched - rel, datavalues', rel, datavalues)
                    yield attr, datavalues

    rel_datavalues_matched_tuples = list(_get_value_from_attr_and_wd_item(wd_item, attr))
    return rel_datavalues_matched_tuples