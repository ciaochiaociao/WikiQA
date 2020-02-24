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
    if attr == '名字':
        aliases = get_all_aliases_from_dict(wd_item)
        return aliases[0] + aliases[1]
    for rel, datavalues in wd_item['claims'].items():
        labels, _ = rel  # only use label to match
        for label in labels:
            if label == attr:
                return datavalues


def postprocess_datavalue(datavalue):
    datatype = get_datatype(datavalue)
    processed_list: List[str] = postprocess(datavalue, datatype)
    return processed_list


def get_datatype(datavalue):
    if isinstance(datavalue, str):
        datatype = 'str'
    else:
        datatype = datavalue['type']
    return datatype


class WDValue:

    def __init__(self, type_, value):
        self.type = type_  # WDItem, time/date, quantity, string
        self.value = ...


class WDItem:

    def __init__(self, id_, type_):
        self.id = id_
        self.type = ...
        self.claims: List[WDClaim] = ...

        self.has_qualifiers = ...
        if self.has_qualifiers:
            self.qualifiers: List[WDClaim] = ...


class WDClaim:

    def __init__(self, property, values):
        self.property: WDProperty = ...
        self.values: List[WDValue] = ...


class WDProperty:

    def __init__(self, id_, labels, aliases):
        self.id_ = ...
        self.labels = ...
        self.aliases = ...


def postprocess(datavalue, datatype):
    if datatype == 'wikibase-item':
        object_aliases = datavalue['all_aliases']  # a tuple
        object_aliases = tuple([list(set([cc.convert(name) for name in names])) for names in object_aliases])
        postprocessed = [object_aliases]
    elif datatype == 'time':
        sutime_format = r'(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)' \
                        r'T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\.(?P<millisecond>\d+)Z'
        m = re.match(sutime_format, datavalue['value'])
        year, month, day = int(m.groupdict()['year']), int(m.groupdict()['month']), int(m.groupdict()['day'])
        time_formats = [
            '{}年{}月{}日'.format(year, month, day),
            '{}年{}月{}日'.format((int(year) - 1911), month, day),
            '{}年{}月{}號'.format(year, month, day),
            '{}年{}月{}號'.format((int(year) - 1911), month, day),
            '{}年{}月'.format(year, month),
            '{}年{}月'.format((int(year) - 1911), month),
            '{}年'.format(year),
            '{}年'.format((int(year) - 1911))
        ]
        postprocessed = time_formats
    elif datatype == 'quantity':
        amount = datavalue['value']['amount']
        unit_id = datavalue['value']['unit']
        unit_labels, unit_aliases = get_all_aliases_from_id(unit_id)
        postprocessed = [str(amount) + unit_name for unit_name in unit_labels + unit_aliases]
    elif datatype == 'str':
        postprocessed = [datavalue]
    else:
        object_value = datavalue['value']
        postprocessed = [object_value]

    return postprocessed


