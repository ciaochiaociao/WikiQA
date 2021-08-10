from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

with open('../data/raw/1.7.13/FGC_release_all_test.json', encoding='utf-8') as f:
	docs = json.load(f)

wiki_json_qa = WikiQA(corenlp_ip='http://140.109.19.51:9000',
                 wikidata_ip='mongodb://140.109.19.51:27020',
		       pred_infer='rule',
		       mode='prod',
                verbose=False)

wiki_qa = WikiQA(corenlp_ip='http://140.109.19.51:9000',
                    wikidata_ip='mongodb://140.109.19.51:27020',
                    pred_infer='neural',
					neural_model_fpath = 'files/v2_with_nones_early_stop.h5',
					dataset_fpath = 'files/predicate_inference_combined_v2_with_nones.csv',
					tokenizer = 'files/bert-base-chinese-vocab.txt',
                    mode='prod')

for doc in docs:
	all_answers = wiki_json_qa.predict_on_qs_of_one_doc(doc)
	print('(rule) output:')
	pprint(all_answers)

	all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
	print('(neural) output:')
	pprint(all_answers)
