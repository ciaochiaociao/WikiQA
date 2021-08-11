from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

q = '苏东坡的爸爸叫什么名字?'

CORENLP_IP = 'http://140.109.19.191:9000'
MONGODB_IP = 'mongodb://140.109.19.51:27020'

wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                wikidata_ip=MONGODB_IP,
                pred_infer='rule',
                mode='prod',
                check_atype_w_NE_type=False,
                match_in_passage=False,
                use_NE_span_in_psg=False,
                use_se='None',
                verbose=True)
all_answers = wiki_qa.predict_on_q(q)
pprint(all_answers)
