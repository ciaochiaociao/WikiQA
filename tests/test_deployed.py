from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

with open('../data/raw/predict/question_result.json', encoding='utf-8') as f:
	docs = json.load(f)

CORENLP_IP = 'http://localhost:9000'

for doc in docs:

	wiki_qa = WikiQA(CORENLP_IP)
	all_answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=True)  # use data[0:2] to just answer the first two passages for the pilot run
	print('(USE JSON) output:')
	pprint(all_answers)

	all_answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=False, use_se='pred')
	print('(WIKIQA) output:')
	pprint(all_answers)
