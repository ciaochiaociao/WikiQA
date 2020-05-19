# WikiQA
## Requirements
stanfordnlp
python-dotenv
ansi

## Usage

```python
from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

with open('FGC_release_all(cn).json', encoding='utf-8') as f:
	docs = json.load(f)

CORENLP_IP = 'http://localhost:9000'

wiki_qa = WikiQA(CORENLP_IP)
for doc in docs:
    all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
    #same as below
    #all_answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=True, use_se='pred', neural_pred_infer=False)
    
    pprint(all_answers)
```
```
#example input
{	'DID': 'D001',
	'DTEXT': ...,
	'QUESTIONS': [
		{'QID': 'D001Q01', 'QTEXT': ...},
		{'QID': 'D001Q02', 'QTEXT': ...},
		{'QID': 'D001Q03', 'QTEXT': ...},
		...
	],
	...
}
```
```
#example output
[[{'AMODULE': 'WikiQA',
   'ATEXT': '北宋',
#   'QID': 'D001Q01',
   'SCORE': 1,
   'SCORE_E': 0,
   'SCORE_S': 0}],
 [{'AMODULE': 'WikiQA',
   'ATEXT': '苏洵',
#   'QID': 'D001Q03',
   'SCORE': 1,
   'SCORE_E': 0,
   'SCORE_S': 0}],
 [{'AMODULE': 'WikiQA',
   'ATEXT': '唐代',
#   'QID': 'D001Q06',
   'SCORE': 1,
   'SCORE_E': 0,
   'SCORE_S': 0}],
 [{'AMODULE': 'WikiQA',
   'ATEXT': '宋代',
#   'QID': 'D001Q09',
   'SCORE': 1,
   'SCORE_E': 0,
   'SCORE_S': 0}]]
```

## Release Notes

* 0.1 - Original Release  
* 0.2 - Add fgc_knowledgebase.json, fgc_kb mode  
* 0.3 - Refactor WikiQA - 1  
* 0.4 - Refactor WikiQA - 2  
* 0.5 - Refactor WikiQA - 3  
* 1.0 - Add Supporting Evidence and Neural Predicate Inference, ...  
* 1.0.2 - Modify Supporting Evidence, ATYPE, AMODE attribute names in json during prediction  
* 1.0.3   
    - Modify Supporting Evidence, ATYPE, AMODE attribute names in json during prediction Again
    - Add cached wikidata_utils function 
    
* 2.0  

    Key Features:
    - Add 高度, 墓地 and custom 寿命, 成立或建立年份 attributes in parsing question
    - Fix Entity linking by querying also traditional chinese by converting from simplified one with OpenCC
    - Facilitate Attribute 名字 Parsing by adding filtering rule that get rids of the answer that has common words with the question 
    
    Others:
    - Add `WikiQAConfig` class
    - Adapt to dataset format of the version 1.7.13
      AMODE, ATYPE, SHINT, AMODE_, ATYPE_, SHINT_
        
    Overall Performance:
    32 / 34 = 94.1%