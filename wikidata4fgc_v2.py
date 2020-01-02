from pymongo import MongoClient
from typing import Dict, List
from opencc import OpenCC

cc3 = OpenCC('s2t')
cc2 = OpenCC('s2tw')
cc = OpenCC('s2twp')
host = '140.109.19.51'
port = '27020'

client = MongoClient('mongodb://{}:{}'.format(host, port))


def get_multiling_label_from_pid(pid: int) -> Dict[str, str]:
    results = list(client.zhwiki.properties.find({'id':pid}))

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['labels']}


def get_multiling_aliases_from_pid(pid: int):
    results = list(client.zhwiki.properties.find({'id':pid}))

    return {lang: r['aliases'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['aliases']}


def get_multiling_label_from_id(id_: int):
    results = list(client.zhwiki.entities.find({'id':id_}))

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['labels']}


def get_multiling_aliases_from_id(id_: int):
    results = list(client.zhwiki.entities.find({'id':id_}))

    return {lang: r['labels'][lang] for r in results for lang in ['zh', 'zh-tw', 'zh-cn', 'en'] if lang in r['aliases']}


def get_all_aliases_from_id(id_: int):
    labels = get_multiling_label_from_id(id_)
    aliases = get_multiling_aliases_from_id(id_)
    return get_all_from_labels_aliases(labels, aliases)


def get_all_aliases_from_pid(pid: int):
    labels = get_multiling_label_from_pid(pid)
    aliases = get_multiling_aliases_from_pid(pid)
    return get_all_from_labels_aliases(labels, aliases)


def get_all_aliases_from_dict(dict_):

    return get_all_from_labels_aliases(dict_['labels'], dict_['aliases'])


def get_all_from_labels_aliases(labels, aliases):
    
    def to_zh_tw(dict_, cc):
        results = set()
        for locale, label in dict_.items():
            if locale in ['zh-cn', 'zh']:
                if isinstance(label, list):
                    for item in label:
                        newlabel = cc.convert(item)
                        results.add(newlabel)
                else:
                    newlabel = cc.convert(label)
                    results.add(newlabel)
            elif locale == 'zh-tw':
                if isinstance(label, list):
                    results.update(label)
                else:
                    results.add(label)
            else:
                pass
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

        
def get_dicts_from_pid(id_: int):
    """
    >>> get_dicts_from_pid('P31')[0]['labels']['zh']
    '苏轼'
    """
    return list(client.zhwiki.properties.find({'id':id_}))


def get_dicts_from_id(id_: int):
    """
    >>> get_dicts_from_id('Q36020')[0]['labels']['zh']
    '苏轼'
    """
    return list(client.zhwiki.entities.find({'id':id_}))


def get_dicts_from_sitelink_title(sitelink_title):
    """
    >>> get_qid_from_sitelink_title('苏轼')['id']
    'Q36020'
    """
    return list(client.zhwiki.entities.find({'sitelinks.zhwiki':sitelink_title}))

    
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
        results = list(client.zhwiki.entities.find({'labels.{}'.format(locale): keyword}))
        results2 = list(client.zhwiki.entities.find({'aliases.{}'.format(locale):keyword}))
        for result in result0 + results + results2:
            if result['id'] not in all_ids:
                all_results.append(result)
                all_ids.append(result['id'])
                
    return all_results


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
            rule1 = lambda labels: 'ID' in labels['en']
            rule2 = lambda labels: labels['en'] in \
                ['ISNI', 'image', 'Commons Creator page', 'National Library of Korea Identifier', 'URI', 'Libris-URI', \
                 "topic's main template", 'British Museum person-institution', 'NLC authorities', 'Commons category']
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


if __name__ == '__main__':
    from pprint import pprint
#     print("run test: get_dicts_from_keyword('東坡居士')")
    for d in get_dicts_from_keyword('東坡居士'):
        pprint(get_all_aliases_from_dict(d))
#     pprint(get_multiling_label_from_id('Q36020'))

    for d in get_dicts_from_id('Q36020'):
        d = filter_claims_in_dict(d)
        pprint(readable(d))
