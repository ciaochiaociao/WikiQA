from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

with open('../data/raw/1.7.13/FGC_release_all_test.json', encoding='utf-8') as f:
	docs = json.load(f)

wiki_json_qa = WikiQA(corenlp_ip='http://140.109.19.51:9000',
                 wikidata_ip='mongodb://140.109.19.51:27020',
                 use_fgc_kb=True,
		       pred_infer='rule',
		       mode='prod',
                verbose=False)

wiki_qa = WikiQA(corenlp_ip='http://140.109.19.51:9000',
                    wikidata_ip='mongodb://140.109.19.51:27020',
                    use_fgc_kb=False,
                    pred_infer='rule',
                    mode='prod')
for doc in docs:

	all_answers = wiki_json_qa.predict_on_qs_of_one_doc(doc)
	print('(USE JSON) output:')
	pprint(all_answers)

	all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
	print('(WIKIQA) output:')
	pprint(all_answers)
