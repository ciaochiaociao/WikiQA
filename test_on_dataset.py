from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

with open('data/processed/1.7.13/FGC_release_all_dev_filtered.json', encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(corenlp_ip='http://140.109.19.191:9000',
                wikidata_ip='mongodb://140.109.19.51:27020',
                pred_infer='rule',
                mode='prod',
                verbose=False)
for doc in docs:
    all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
    if all_answers[0]['ATEXT']:
        pprint(all_answers)
        break
