# WikiQA
## Requirements

1. Python
```
opencc-python-reimplemented
pymongo
stanfordnlp
ansi
tensorflow  # only needed if you want to use neural model
regex
tensorflow-gpu
pandas
numpy
pymongo
click
transformers==4.9.2
```
2. Wikidata Mongo Database
3. CoreNLP >= 3.9.2

## Installation
 - `conda env create` at the project folder which will read the environment.yml file
 - `pip install -r requirements.txt`
 - `scp -r cwhsu@140.109.19.51:~/workspace/FGC/WIKIKB4FGC/mongodb_dbpath_wiki_zh > /path/to/mongodbdata`
 - changing `/path/to/mongodbdata` in `docker-compose.yml` file
 - `sudo docker-compose up  # for setting up Stanford CoreNLP, Wikidata Monogo Database`

## Production
### Folder Structure

```
fgc_wiki_qa                                           
├── __init__.py                                       
├── commands                                          
│   ├── __init__.py                                   
│   ├── evaluate.py                                   
│   └── run_on_fgc.py                                 
├── config.py                                         
├── data                                              
│   ├── __init__.py                                   
│   ├── filter_dataset.py                             
│   ├── get_qa_tsv.py                                 
│   ├── get_toy_dataset.py                            
│   └── merge.py                                      
├── files                                             
│   ├── bert-base-chinese-vocab.txt                   
│   ├── predicate_inference_combined_v2_with_nones.csv
│   └── v2_with_nones_early_stop                   
├── libs                                              
│   └── huggingface_utils                             
│       ├── example.ipynb                             
│       ├── huggingface_utils.py                      
│       └── requirements.txt                          
├── manual_data                                       
│   ├── __init__.py                                   
│   └── get_dataset.py                                
├── metrics                                           
│   ├── __init__.py                                   
│   ├── analyze_qids.py                               
│   ├── error_analysis.py                             
│   └── evaluation.py                                 
├── models                                            
│   ├── __init__.py                                   
│   ├── entity_linking.py                             
│   ├── neural_predicate_inference                    
│   │   ├── __init__.py                               
│   │   ├── predict.py                                
│   │   ├── predict_on_FGC.py                         
│   │   └── result.xlsx                               
│   ├── predicate_inference_neural.py                 
│   ├── predicate_inference_regex.py                  
│   ├── predicate_inference_rules.py                  
│   ├── value2ans.py                                  
│   └── wikiqa.py                                     
├── utils                                             
│   ├── __init__.py                                   
│   ├── fgc_utils.py                                  
│   ├── fuzzy_match.py                                
│   ├── stanfordnlp_utils.py                          
│   ├── utils.py                                      
│   ├── wikidata4fgc.py                               
│   └── wikidata_utils.py                             
├── README.md
└── requirements.txt
```

### Usage


```python
from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

doc = { 'DID': 'D001',
    'DTEXT': '苏轼（1037年1月8日－1101年8月24日），眉州眉山（今四川省眉山市）人，北宋时著名的文学家、政治家、艺术家、医学家。字子瞻，一字和仲，号东坡居士、铁冠道人。嘉佑二年进士，累官至端明殿学士兼翰林学士，礼>
    'QUESTIONS': [
        {'QID': 'D001Q01', 'QTEXT': '苏东坡的爸爸叫什么名字?'},
    ],
}

CORENLP_IP = 'http://localhost:9000'
MONGODB_IP = 'mongodb://140.109.19.51:27020'

wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                wikidata_ip=MONGODB_IP,
                pred_infer='rule',
                mode='prod',
                verbose=False)

all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
pprint(all_answers)
```
arguments:
 - pred_infer: `rule`/`neural` method to infer predicate (default to rule, which has better precision so far)
 - mode: `prod`/`dev` (e.g. `dev` has access to the gold answer)
 - verbose: show verbose message during prediction

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

## Development

To use the tools for development, `make` is required. 
(can be installed by `sudo apt-get install build-essential`)

### Folder Structure
```
.
├── data
│   ├── download
│   │   ├── 1.7.9.7z
│   │   ├── 4.2-1.7z
│   │   ├── 4.4-1.7z
│   │   ├── FGC1.7.8-revise-sp.zip
│   │   ├── FGC_releasae_1.7.12.7z
│   │   ├── FGC_release_1.7.11.7z
│   │   └── FGC_release_1.7.13.7z
│   ├── external
│   │   ├── error_ids_by_module.json
│   │   ├── fgc_knowledgebase.json
│   │   ├── fgc_predicate_inference_v0.1.tsv
│   │   ├── fgc_wiki.csv
│   │   ├── fgc_wiki_benchmark_v0.1.tsv
│   │   └── manual_after_auto_dataset.xlsx
│   ├── processed
│   │   ├── 1.7.13
│   │   │   ├── FGC_release_all_dev_filtered.json
│   │   │   ├── FGC_release_all_test_filtered.json
│   │   │   ├── FGC_release_all_train_filtered.json
│   │   │   ├── merged_filtered.json
│   │   │   ├── qa_all.tsv
│   │   │   ├── qa_dev.tsv
│   │   │   ├── qa_test.tsv
│   │   │   ├── qa_train.tsv
│   │   │   └── toy_all.json
│   │   └── ...
│   └── raw
│       ├── 1.7.13
│       │   ├── FGC_release_all_dev.json
│       │   ├── FGC_release_all_test.json
│       │   ├── FGC_release_all_train.json
│       │   ├── FGC_release_ss_test.json
│       │   ├── merged.json
│       │   └── qa_all.tsv
│       ├── ...
│       └── predict
│           └── question_result.json
├── docker-compose.yml
├── experiments
│   ├── v2.0.3_on_1.7.13
│   │   ├── config.json
│   │   ├── error_analysis_all.xlsx
│   │   ├── file4eval_all.tsv
│   │   ├── qids_all.json
│   │   ├── report_all.txt
│   │   ├── run.log
│   │   ├── v2.0.3_on_1.7.13_VS_v2.0.2_on_1.7.13.json
│   │   └── v2.0.3_on_1.7.13_VS_v2.0.2_on_1.7.13.txt
│   └── ...
├── fgc_wiki_qa
│   └── ...
├── models
│   └── neural_predicate_inference
│       ├── bert-base-chinese-vocab.txt
│       ├── predicate_inference_combined_v2_with_nones.csv
│       └── v2_with_nones_early_stop.h5
├── notebooks  # mostly for visualization
│   ├── parse_questions.ipynb
│   ├── process_fgc_wiki_dataset.ipynb
│   ├── read_dataset.ipynb
│   ├── tokenizer.ipynb
│   ├── visual_whole_system_performance.ipynb
│   └── visualize.ipynb
├── released
│   ├── WIKIQA_V1.0.1.tar.gz
│   └── ...
├── reports
│   ├── config.json
│   ├── file4eval.tsv
│   └── run.log
├── tests
│   ├── test_deployed.py
│   └── test_predicate_inference_neural.py
├── release.sh
├── create_fgc_pred_infer_dataset.sh
├── run_exp.sh
│── Makefile
├── requirements.txt
└── README.md
```

### data pipeline
```
# download
make get FGC_DATASET_7z_URL=... FGC_VER=...
# data preprocessing
make all FGC_VER=...
```

### run experiment
```
export EXP_DIR=experiments/v2.0_on_1.7.12  # output directory path
export PRED_INFER=rule  # `rule`/`pred` predicate inference method
export FGC_VER=1.7.12  # dataset version to run prediction on
export RUN_ON=raw  # run on the `raw` or `proc` version of the dataset
export USE_SE=pred  # `pred` or `gold`
make run exp
```

### Others
```
# compare performance between two experiments
make compare EXP_DIR=... COMPARED_EXP_DIR=...
```

### Release/Deployment

```
#release for deployment to the whole system
bash ./release.sh <ver_num>  # do testing, compressing content in `fgc_wiki_qa` folder to a file, and move the compressed file to `release` folder
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
