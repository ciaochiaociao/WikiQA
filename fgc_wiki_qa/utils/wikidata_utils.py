#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

# TODO: add dependency injection (specifically, add a class that can be instantiated with IP address)
# TODO: move configuration information, e.g., host and port, to .env file

from pymongo import MongoClient
from typing import Dict, List, Union
from opencc import OpenCC
from functools import lru_cache
import logging
import re

cc = OpenCC('t2s')
cc2 = OpenCC('tw2s')
cc3 = OpenCC('tw2sp')
cc_s2t = OpenCC('s2t')
host = '140.109.19.51'  # localhost
port = '27020'

QUERY_ON_DEMAND = True

client = MongoClient('mongodb://{}:{}'.format(host, port))

CACHE_SIZE = 1024 * 2  # default is only 128


@lru_cache(CACHE_SIZE)
def get_dicts_from_pid(id_: str):
    """
    >>> get_dicts_from_pid('P31')[0]['labels']['zh']
    '苏轼'
    """
    return list(client.zhwiki.properties.find({'id': id_}))


@lru_cache(CACHE_SIZE)
def get_dicts_from_id(id_: str):
    """
    >>> get_dicts_from_id('Q36020')[0]['labels']['zh']
    '苏轼'
    """
    return list(client.zhwiki.entities.find({'id': id_}))


@lru_cache(CACHE_SIZE)
def get_dicts_from_sitelink_title(sitelink_title):
    """
    >>> get_dicts_from_sitelink_title('苏轼')[0]['id']
    'Q36020'
    """
    return list(client.zhwiki.entities.find({'sitelinks.zhwiki': sitelink_title}))


# @lru_cache(CACHE_SIZE)
def get_dicts_from_labels(locale, keyword):
    dicts = []
    for keyword in [_cc.convert(keyword) for _cc in [cc, cc_s2t]]:
        dicts.extend(_get_dicts_from_labels(locale, keyword))
    return dicts


# @lru_cache(CACHE_SIZE)
def _get_dicts_from_labels(locale, keyword):
    return list(client.zhwiki.entities.find({'labels.{}'.format(locale): keyword}))


# @lru_cache(CACHE_SIZE)
def get_dicts_from_aliases(locale, keyword):
    dicts = []
    for keyword in [_cc.convert(keyword) for _cc in [cc, cc_s2t]]:
        dicts.extend(_get_dicts_from_aliases(locale, keyword))
    return dicts


def _get_dicts_from_aliases(locale, keyword):
    return list(client.zhwiki.entities.find({'aliases.{}'.format(locale): keyword}))


CACHED_FUNCS = [get_dicts_from_pid, get_dicts_from_id, get_dicts_from_sitelink_title, get_dicts_from_labels,
                get_dicts_from_aliases]


def get_cache_info():
    return {
        'get_dicts_from_pid': get_dicts_from_pid.cache_info(),
        'get_dicts_from_id': get_dicts_from_id.cache_info(),
        'get_dicts_from_sitelink_title': get_dicts_from_sitelink_title.cache_info(),
        'get_dicts_from_labels': get_dicts_from_labels.cache_info(),
        'get_dicts_from_aliases': get_dicts_from_aliases.cache_info(),
    }


def get_multiling_label_from_pid(pid: str) -> Dict[str, str]:
    results = get_dicts_from_pid(pid)

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['labels']}


def get_multiling_aliases_from_pid(pid: str):
    results = get_dicts_from_pid(pid)

    return {lang: r['aliases'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if
            lang in r['aliases']}


def get_multiling_label_from_id(id_: str):
    results = get_dicts_from_id(id_)

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['labels']}


def get_multiling_aliases_from_id(id_: str):
    results = get_dicts_from_id(id_)

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['aliases']}


def get_all_aliases_from_id(id_: str):
    labels = get_multiling_label_from_id(id_)
    aliases = get_multiling_aliases_from_id(id_)
    return get_all_from_labels_aliases(labels, aliases)


def get_all_aliases_from_pid(pid: str):
    labels = get_multiling_label_from_pid(pid)
    aliases = get_multiling_aliases_from_pid(pid)
    return get_all_from_labels_aliases(labels, aliases)


def get_all_aliases_from_dict(dict_):
    return get_all_from_labels_aliases(dict_['labels'], dict_['aliases'])


def get_all_from_labels_aliases(labels, aliases):
    def to_zh_tw(dict_, cc):
        results = set()
        for locale, label in dict_.items():
            if isinstance(label, list):
                for item in label:
                    newlabel = cc.convert(item)
                    results.add(newlabel)
            else:
                newlabel = cc.convert(label)
                results.add(newlabel)
        return results

    labels_zh_tw = to_zh_tw(labels, cc) | to_zh_tw(labels, cc2) | to_zh_tw(labels, cc3)
    aliases_zh_tw = to_zh_tw(aliases, cc) | to_zh_tw(aliases, cc2) | to_zh_tw(aliases, cc3)
    aliases_zh_tw = [alias for alias in aliases_zh_tw if alias not in labels_zh_tw]
    return (tuple(labels_zh_tw), tuple(aliases_zh_tw))


def get_fallback_zh_from_dict(d):
    for lang in ['zh-tw', 'zh', 'zh-cn']:
        try:
            return d[lang]
        except:
            pass


def get_fallback_zh_label_from_dict(d):
    return get_fallback_zh_from_dict(d['labels'])
    #     for lang in ['zh-tw', 'zh', 'zh-cn']:


#         try:
#             return d['labels'][lang]
#         except:
#             pass

def get_dicts_from_keyword(keyword):
    """
    >>> get_dicts_from_keyword('苏轼')
    'Q36020'
    """
    locales = ['zh', 'zh-tw', 'zh-cn']
    all_results = []
    all_ids = []

    result0 = get_dicts_from_sitelink_title(keyword)
    for locale in locales:
        results = get_dicts_from_labels(locale, keyword)
        results2 = get_dicts_from_aliases(locale, keyword)
        for result in result0 + results + results2:
            if result['id'] not in all_ids:
                all_results.append(result)
                all_ids.append(result['id'])

    return all_results


def get_pid_from_name(name):
    locales = ['zh', 'zh-tw', 'zh-cn']
    for locale in locales:
        res = get_dicts_from_labels(locale, name)
        if res:
            return res


def filter_claims_in_dict(d):
    from copy import deepcopy

    new_d = deepcopy(d)
    new_results = []

    def claims_filter(claims_dict):
        results = {}

        def filtering_rules(pid, pds):
            labels = get_multiling_label_from_pid(pid)
            # rule1: filter out the claims with english names having 'ID string'
            # rule2: filter out the claims with english names in the following list
            # rule3: filter out the claims without any chinese language labels
            rule1 = lambda labels: 'ID' in labels.get('en', '')
            rule2 = lambda labels: labels.get('en', '') in \
                                   ['ISNI', 'image', 'Commons Creator page', 'National Library of Korea Identifier',
                                    'URI', 'Libris-URI', \
                                    "topic's main template", 'British Museum person-institution', 'NLC authorities',
                                    'Commons category']
            rule3 = lambda labels: not any([lang in labels for lang in ['zh', 'zh-cn', 'zh-tw']])
            if rule1(labels) or rule2(labels) or rule3(labels):
                return False
            return True

        for pid, pds in claims_dict.items():
            #             print(pid)
            if filtering_rules(pid, pds):
                npds = []
                for pd in pds:  # rule4: datatype must not be 'external-id', 'commonsMedia'
                    if pd['type'] not in ['external-id', 'commonsMedia', 'tabular-data', 'math', 'musical-notation']:
                        npds.append(pd)

                    # rule5: qualifiers
                    if 'qualifiers' in pd:
                        new_qualifiers = claims_filter(pd['qualifiers'])
                        if len(new_qualifiers):
                            pd['qualifiers'] = new_qualifiers
                        else:
                            del pd['qualifiers']

                if len(npds):  # rule5: only keep the nonempty relations
                    results.update({pid: npds})

        return results

    props = claims_filter(d['claims'])
    new_d['claims'] = props
    return new_d


def readable(d: Dict):
    def _convert(_d, props_key):
        nd = {}
        for pid, pds in _d[props_key].items():
            try:
                #                 nk = get_multiling_label_from_pid(pid)['en']
                #                 nk = get_fallback_zh_from_dict(get_multiling_label_from_pid(pid))
                #                 nk = tuple(get_multiling_label_from_pid(pid).items())
                nk = tuple(get_all_aliases_from_pid(pid))
                nd[nk] = _d[props_key].get(pid)
            except KeyError:
                #                 print(_d)
                #                 print(props_key)
                #                 print(pid, pds)
                pass

            for pd in pds:
                # translate QID, PID to its value
                if pd['type'] == 'wikibase-item':
                    if 'value' in pd:
                        #                         pd['label'] = get_multiling_label_from_id(pd['value'])
                        dicts_ = get_dicts_from_id(pd['value'])
                        try:
                            pd['all_aliases'] = get_all_aliases_from_dict(dicts_[0])
                        except KeyError:
                            pd['all_aliases'] = None
                    else:
                        pd['all_aliases'] = None
                elif pd['type'] == 'wikibase-property':
                    pd['label'] = get_all_aliases_from_pid(pd['value'])
                else:
                    pass

                try:
                    pd['qualifiers'] = _convert(pd, 'qualifiers')
                except KeyError:
                    pass

        return nd

    d['claims'] = _convert(d, 'claims')

    return d


def clean_and_simplify_wd_items(all_wd_items):
    all_wd_items = map(filter_claims_in_dict, all_wd_items)
    all_wd_items = map(readable, all_wd_items)
    return list(all_wd_items)


def get_datatype(datavalue):
    if isinstance(datavalue, str):
        datatype = 'str'
    else:
        datatype = datavalue['type']
    return datatype


# --------------------------------------------------------------------
# NEW API (V3)
def require_query(f):
    def _require_query(*args, **kwargs):
        self_ = args[0]
        if self_.status == 'queried':
            return f(*args, **kwargs)
        else:
            if QUERY_ON_DEMAND:
                self_.query()
                return f(*args, **kwargs)
            else:
                return

    return _require_query


def queryable(query_func):
    def wrapper(*args, **kwargs):
        self_ = args[0]
        force = kwargs.get('force')
        #         print('status:', self_.status)
        if self_.status != 'queried' or force:
            try:
                #                 print('querying ...')
                query_func(*args, **kwargs)
                self_.status = 'queried'
                code = 'success'
            #                 print('status:', self_.status)
            except Exception as e:
                self_.status = 'query_failed'
                code = 'fail'
                print(e)
        else:
            code = 'already_queried'
            print('Already queried.')
        return code

    return wrapper


def get_items_from_keyword(keyword, greedy_lvl=0):
    ds = get_dicts_from_keyword(keyword)
    wd_items = []
    for d in ds:
        wd_items.append(parse_item_greedy(d, greedy_lvl))
    return wd_items


class WDItem:

    def __init__(self, id_, labels, aliases, claims, dict_):
        self.id = id_
        self.labels: WDMLLabel = labels
        self.aliases: WDMLAliases = aliases

        self.claims: List[WDClaim] = claims
        self.dict = dict_
        self.status = 'queried'

    def __repr__(self):
        return f'({self.id}) {self.__str__()}'

    def __str__(self):
        return repr(self.labels) or repr(self.aliases)

    @property
    def is_valid(self):
        return self.labels.is_valid or self.aliases.is_valid

    @property
    def valid_claims(self):
        return [claim for claim in self.claims if claim.is_valid]

    # @property
    # def triples(self):
    #     return self._get_triples(self.claims)
    #
    # def _get_triples(self, claims, join_values: Optional[str] = None, subj=None, pred=None, **kwargs):
    #     if subj is None:
    #         subj = self.__str__()
    #
    #     if pred is None:
    #         pred = str(claim.property)
    #
    #     if join_values is None:
    #         return [(subj, pred, str(value)) for claim in claims for value in claim.values]
    #     else:
    #         return [(subj, pred, claim.get_joined_values(join=join_values, **kwargs)) for claim in claims]
    #
    # @property
    # def valid_triples(self):
    #     return self._get_triples(self.valid_claims)

    @require_query
    def traverse(self, ix):
        return self.claims[ix].values

    def get_all_aliases(self, min_len=1):
        all_aliases = []
        if self.labels.zh_names:
            all_aliases.extend(self.labels.zh_names)
        if self.aliases.zh_names:
            all_aliases.extend(self.aliases.zh_names)
        return list(set(filter(lambda name: len(name) > min_len, all_aliases)))


class WDValue:

    def __init__(self, type_, qualifiers):
        self.qualifiers: List[WDClaim] = qualifiers
        self.type = type_

    def get_all_aliases(self):
        # default
        return [self.__str__()]


class WDItemValue(WDItem, WDValue):

    def __init__(self, type_, id_, qualifiers, greedy=False):
        WDValue.__init__(self, type_, qualifiers)
        WDItem.__init__(self, id_, None, None, None, None)
        self.status = 'unqueried'
        if greedy:
            self.query()

    @require_query
    def __repr__(self):
        return WDItem.__repr__(self)

    @require_query
    def __str__(self):
        return WDItem.__str__(self)

    @queryable
    def query(self, force=False) -> None:
        self.dict = get_dicts_from_id(self.id)[0]
        wd_item = parse_item(self.dict)
        self.labels = wd_item.labels
        self.aliases = wd_item.aliases
        self.claims = wd_item.claims


class WDClaim:

    def __init__(self, prop, values):
        self.property: WDProperty = prop
        self._values: List[Union[WDItem, WDTime, WDQuantity, str]] = values

    def __repr__(self):
        return f'[{repr(self.property)}] {", ".join([repr(value) for value in self.values])}'

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = [value for value in values if value.is_valid]

    def get_joined_values(self, join: str = '、', filter_func=lambda x: True):
        return join.join([str(value) for value in filter(filter_func, self.values) if str(value)])

    @property
    def is_valid(self):
        return self.property.is_valid


class WDProperty:

    def __init__(self, id_, greedy=False):
        self.id = id_
        self.labels: WDMLLabel = None
        self.aliases: WDMLAliases = None
        self.dict = None
        self.status = 'unqueried'
        if greedy:
            self.query()

    @queryable
    def query(self, force=False):
        self.dict = get_dicts_from_pid(self.id)[0]
        self.labels = parse_ml_labels(self.dict['labels'])
        self.aliases = parse_ml_aliases(self.dict['aliases'])

    @require_query
    def __repr__(self):
        return repr(self.labels) or repr(self.aliases)

    @require_query
    @property
    def is_valid(self):
        return self.labels.is_valid or self.aliases.is_valid

    @require_query
    def get_all_aliases(self, min_len=1):
        all_aliases = []
        if self.labels:
            all_aliases.extend(self.labels.zh_names)
        if self.aliases.zh_names:
            all_aliases.extend(self.labels.zh_names)
        return list(filter(lambda name: len(name) > min_len, all_aliases))


class WDLabel:

    def __init__(self, lang, name):
        self.lang = lang
        self.name = name


class WDAliases:

    def __init__(self, lang, names: list):
        self.lang = lang
        self.names = names


class Multilingual:

    def __repr__(self):
        return self.main_zh_name

    def fallback_zh_names(self):
        raise NotImplementedError

    def zh_names(self) -> list:
        raise NotImplementedError

    def fallback_zh_names(self):
        return list(sorted(self.zh_names, key=len, reverse=True))

    @property
    def main_zh_name(self) -> str:
        try:
            return self.fallback_zh_names()[0]
        except IndexError:
            return ''

    @property
    def is_valid(self):
        return bool(self.main_zh_name)


class WDMLLabel(Multilingual):

    def __init__(self, ml_label):
        self.ml_label: List[WDLabel] = ml_label

    @property
    def zh_names(self):
        LANGS = ['zh', 'zh-tw', 'zh-cn']
        return list(set([cc.convert(label.name) for label in self.ml_label if label.lang in LANGS]))


class WDMLAliases(Multilingual):

    def __init__(self, ml_aliases):
        self.ml_aliases: List[WDAliases] = ml_aliases

    @property
    def zh_names(self):
        LANGS = ['zh', 'zh-tw', 'zh-cn']
        return list(
            set([cc.convert(alias) for aliases in self.ml_aliases if aliases.lang in LANGS for alias in aliases.names]))


# ref: https://www.wikidata.org/wiki/Category:Properties_by_datatype
# ref2: https://www.mediawiki.org/wiki/Wikibase/DataModel#Datatypes_and_their_Values
# 17 wikidata value types
# globe-coordinate, external-id, wikibase-form, geo-shape, wikibase-item, wikibase-lexeme, commonsMedia, monolingualtext, musical-notation
# math, wikibase-property, quantity, wikibase-sense, string, tabular-data, time, url

# interested 6 types: wikibase_item, time, quantity, string, monolingualtext, wikibase-property
class WDQuantity(WDValue):

    def __init__(self, type_, number, unit, qualifiers):
        self.type = type_
        self.number = number
        self.unit = unit
        super().__init__(type_, qualifiers)

    def __repr__(self):
        return str(self.number) + ' ' + str(self.unit)

    @property
    def is_valid(self):
        return bool(self.number) and self.unit.is_valid()


class WDUnit(WDItemValue):

    def __init__(self, unit_id, qualifiers):
        super().__init__('unit', unit_id, qualifiers)

    def __repr__(self):
        if self.id == '1':
            return ''
        else:
            return super().__repr__()


class WDTime(WDValue):
    UNITS = {
        'zh-tw': ['年', '月', '日', '時', '分', '秒', '毫秒'],
        'zh-cn': ['年', '月', '日', '时', '分', '秒', '毫秒'],
        'en': ['Y', 'M', 'D', 'h', 'm', 's', 'ms'],
    }

    def __init__(self, type_, year=None, month=None, day=None, hours=None, minutes=None, seconds=None,
                 milliseconds=None, timezone=None, precision=None, qualifiers=None):
        self.type = type_
        self.year = year
        self.month = month
        self.day = day
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.milliseconds = milliseconds
        self.time_array = [self.year, self.month, self.day, self.hours, self.minutes, self.seconds, self.milliseconds]
        self.timezone = timezone
        self.precision = precision
        super().__init__(type_, qualifiers)

    def __repr__(self):
        return self.to_str('zh-cn', 0, 3)  # year, month, day

    def to_str(self, lang, max_unit, min_unit):
        sep = ''
        time_array = self.time_array[max_unit:min_unit]
        units = WDTime.UNITS[lang][max_unit:min_unit]
        return sep.join([str(num) + unit for num, unit in zip(time_array, units)])

    @property
    def is_valid(self):
        return bool(self.time_array)


def expand_N_hop(value, N):
    lvl = 0

    def _expand(_value: WDValue, lvl):
        if isinstance(_value, WDItemValue) and lvl <= N:
            _value.query()
            lvl += 1
            #             print('lvl', lvl)
            for ix in range(len(_value.claims)):
                values = _value.traverse(ix)
                for __value in values:
                    _expand(__value, lvl)
        else:
            return

    return _expand(value, lvl)


def parse_item_greedy(item_dict, greedy_lvl=0):
    item = parse_item(item_dict)

    for claim in item.claims:
        for value in claim.values:
            expand_N_hop(value, greedy_lvl)

    return item


def parse_item(item_dict):
    id_ = item_dict['id']
    labels = parse_ml_labels(item_dict['labels'])
    aliases = parse_ml_aliases(item_dict['aliases'])
    claims = parse_claims(item_dict['claims'])

    return WDItem(id_, labels, aliases, claims, item_dict)


def parse_claims(claims):
    wd_claims = []
    for pid, values in claims.items():
        wd_claim = parse_claim(pid, values)
        if wd_claim:
            wd_claims.append(wd_claim)
    return wd_claims


def parse_claim(pid, values):
    wd_prop = parse_property(pid)
    wd_values = [parse_value(value) for value in values]
    wd_values = [value for value in wd_values if value is not None]
    if wd_values:
        wd_claim = WDClaim(wd_prop, wd_values)
        return wd_claim


def parse_property(pid):
    prop_dict = get_dicts_from_pid(pid)

    return WDProperty(pid)


def parse_value(value_dict):
    try:
        qualifiers = parse_claims(value_dict['qualifiers'])
    except KeyError:
        qualifiers = None

    type_ = value_dict['type']

    if type_ == 'time':
        return parse_time(value_dict, type_, qualifiers)
    elif type_ == 'quantity':
        return parse_quantity(value_dict, type_, qualifiers)
    elif type_ == 'wikibase-item':
        return parse_item_value(value_dict, type_, qualifiers)
    else:
        return None


def parse_time(time_dict, type_, qualifiers):
    # {'value': '1037-01-08T00:00:00.000Z', 'type': 'time'}
    try:
        sutime = time_dict['value']
        sutime_format = r'(?P<year>[\+\-\d]+)-(?P<month>\d+)-(?P<day>\d+)' \
                        r'T(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(\.(?P<milliseconds>\d+))?Z'
        m = re.match(sutime_format, sutime)
        time_dict = m.groupdict()

    except KeyError:
        logging.error('WDTime instance has no attriubte "value" >> sutime: ' + repr(time_dict))
        return
    except AttributeError:
        logging.error('Parsing time dictionary failed!! >> sutime: ' + sutime)
    except Exception as e:
        logging.error('Unknown Error!! >> sutime: ' + sutime + '\n(Exception):\n' + str(e))

    def _to_int(dict_):
        for k in dict_.keys():
            try:
                dict_[k] = int(dict_[k])
            except TypeError:
                logging.warning(f'No {k} info in sutime >> sutime: ' + sutime)

    _to_int(time_dict)
    return WDTime(type_, qualifiers=qualifiers, **time_dict)


def parse_item_value(item_dict, type_, qualifiers):
    # {'value': 'Q16260880', 'type': 'wikibase-item', 'qualifiers': ...}
    try:
        id_ = item_dict['value']
    except KeyError:
        return

    return WDItemValue(type_, id_, qualifiers)


def parse_quantity(quantity_dict, type_, qualifiers):
    #     {'value': {'amount': 8.8e+23,
    #               'unit': 'Q828224',
    #               'upperBound': None,
    #               'lowerBound': None},
    #      'type': 'quantity'}
    # ref: https://www.wikidata.org/wiki/Wikidata:Units

    try:
        value = quantity_dict['value']
    except KeyError:
        logging.error('Quantity has no value! quantity_dict: ' + repr(quantity_dict))
        return

    # amount
    try:
        amount = value['amount']
    except KeyError:
        logging.error('Quantity has no amount! quantity_dict: ' + repr(quantity_dict))
        return

    try:
        unit_id = value['unit']
    except KeyError:
        logging.error('Quantity has no unit! quantity_dict: ' + repr(quantity_dict))
        return

    qualifiers = value.get('qualifiers', [])

    if not unit_id.startswith('Q') or unit_id != '1':
        logging.error('Unknown unit_id ! quantity_dict: ' + repr(quantity_dict))
        return

    try:
        unit = WDUnit(unit_id)
        if unit.has_names():
            return WDQuantity(type_, amount, unit, qualifiers)
    except Exception:
        logging.error('Other errors happened! quantity_dict: ' + repr(quantity_dict))
        return


def parse_ml_labels(ml_labels_dict):
    return WDMLLabel([WDLabel(lang, name) for lang, name in ml_labels_dict.items()])


def parse_ml_aliases(ml_aliases_dict):
    return WDMLAliases([WDAliases(lang, names) for lang, names in ml_aliases_dict.items()])


if __name__ == '__main__':
    #     from pprint import pprint

    # old api
    #     print("run test: get_dicts_from_keyword('東坡居士')")
    #     for d in get_dicts_from_keyword('東坡居士'):
    #         pprint(get_all_aliases_from_dict(d))
    #     pprint(get_multiling_label_from_id('Q36020'))
    #     print("run test: get_dicts_from_id('Q36020')")
    #     for d in get_dicts_from_id('Q36020'):
    #         d = filter_claims_in_dict(d)
    #         pprint(readable(d))

    # new api
    dicts = get_dicts_from_keyword('東坡居士')
    for dict_ in dicts:
        item = parse_item(dict_)
        print(type(item))
        print(repr(item))
        print(item)

        print(type(item.labels))
        print(repr(item.labels))
        print(item.labels)

        print(type(item.aliases))
        print(repr(item.aliases))
        print(item.aliases)

        print('Ouptutted triples:', item.triples)

        for claim in item.claims:
            print(type(claim))
            print(repr(claim))
            print(claim)

            print(type(claim.property))
            print(repr(claim.property))
            print(claim.property)

            for value in claim.values:
                print(type(value))
                print(repr(value))
                print(value)

                for qualifier in value.qualifiers:
                    print(type(qualifier))
                    print(repr(qualifier))
                    print(qualifier)
                    input()
                input()
            input()
        input()

