# WikiQA
## Requirements
stanfordnlp

## installation
python3 -m pip install stanfordnlp

## Usage
You can run `python3 wikiqa.py` directly or use run the jupyter notebook, `demo.ipynb`

```python
from pprint import pprint
import json
import pandas as pd

with open('FGC_release_all(cn).json', encoding='utf-8') as f:
	docs = json.load(f)

df = pd.DataFrame(data)

wiki_qa = WikiQA()
all_answers = wiki_qa.predict_on_qs_of_one_doc(data)  # use data[0:2] to just answer the first two passages for the pilot run
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

0.1 - Original Release
0.2 - Add fgc_knowledgebase.json, fgc_kb mode
0.3 - Refactor WikiQA - 1 
0.4 - Refactor WikiQA - 2
0.5 - Refactor WikiQA - 3