#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import re
from datetime import timedelta
from typing import List

from .wikidata_utils import get_all_aliases_from_dict, get_all_aliases_from_id, cc, get_datatype
from ..models.predicate_inference_regex import custom


def get_datavalues_from_label(wd_item, attr):
    for rel, datavalues in wd_item['claims'].items():
        labels, _ = rel  # only use label to match
        for label in labels:
            if label == attr:
                return datavalues


def traverse_by_attr_name(wd_item, custom_attr):
    # match predicates
    # new version:
    if custom_attr == '名字':
        aliases = get_all_aliases_from_dict(wd_item)
        return aliases[0] + aliases[1]
    if custom_attr in custom:
        attr = custom[custom_attr]

        if isinstance(attr, str):
            dvalues = get_datavalues_from_label(wd_item, attr)
            if dvalues:
                return dvalues
        else:
            if custom_attr == '寿命':
                dod_attr, dob_attr = attr
                dod_values = get_datavalues_from_label(wd_item, dod_attr)
                dob_values = get_datavalues_from_label(wd_item, dob_attr)
                if dod_values and dob_values:
                    return [(dod_values[0], dob_values[0])]
    else:
        dvalues = get_datavalues_from_label(wd_item, custom_attr)
        if dvalues:
            return dvalues

def postprocess(datavalue, datatype, attr):

    if datatype == 'wikibase-item':
        object_aliases = datavalue['all_aliases']  # a tuple
        postprocessed = [name for names in object_aliases for name in names]
    elif datatype == 'time':
        day, month, year = parse_time(datavalue)
        if '年份' in attr:
            time_formats = [
                '{}年'.format(year),
                '{}年'.format((int(year) - 1911))
            ]
        elif '月份' in attr:
            time_formats = [
                '{}月'.format(month),
            ]
        elif '年月' in attr:
            time_formats = [
                '{}年{}月'.format(year, month),
                '{}年{}月'.format((int(year) - 1911), month)
            ]
        else:
            time_formats = [
                '{}年{}月{}日'.format(year, month, day),
                '{}年{}月{}日'.format((int(year) - 1911), month, day),
                '{}年{}月{}号'.format(year, month, day),
                '{}年{}月{}号'.format((int(year) - 1911), month, day),
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

    # convert Traditional to Simplified
    postprocessed = [cc.convert(v) for v in postprocessed]

    # remove duplicates
    postprocessed = list(dict.fromkeys(postprocessed))

    # remove length one for item except for '朝代' attribute
    if datatype == 'wikibase-item':
        if attr != '朝代':
            postprocessed = [v for v in postprocessed if len(v) > 1]

    return postprocessed


def parse_time(datavalue):
    sutime_format = r'(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)' \
                    r'T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\.(?P<millisecond>\d+)Z'
    m = re.match(sutime_format, datavalue['value'])
    year, month, day = int(m.groupdict()['year']), int(m.groupdict()['month']), int(m.groupdict()['day'])
    return year, month, day


def get_year(life_span: timedelta):
    return life_span.days//365


def postprocess_datavalue(datavalue, attr):
    if attr == '寿命':
        dod, dob = datavalue
        from datetime import date
        life_span = date(*parse_time(dod)) - date(*parse_time(dob))
        life_span_fmt = '{}岁'
        life_span = life_span_fmt.format(get_year(life_span))
        return [life_span]

    datatype = get_datatype(datavalue)
    processed_list: List[str] = postprocess(datavalue, datatype, attr)
    return processed_list