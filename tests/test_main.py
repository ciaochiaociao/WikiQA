#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from unittest import TestCase

from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

CORENLP_IP = 'http://140.109.19.191:9000'
MONGODB_IP = 'mongodb://140.109.19.51:27020'

q = '苏东坡的爸爸叫什么名字?'

p = '苏轼（1037年1月8日－1101年8月24日），眉州眉山（今四川省眉山市）人，北宋时著名的文学家、政治家、艺术家、医学家。字子瞻，一字和仲，号东坡居士、铁冠道人。嘉佑二年进士，累官至端明殿学士兼翰林学士，礼部尚书。南宋理学方炽时，加赐谥号文忠，复追赠太师。有《东坡先生大全集》及《东坡乐府》词集传世，宋人王宗稷收其作品，编有《苏文忠公全集》。其散文、诗、词、赋均有成就，且善书法和绘画，是文学艺术史上的通才，也是公认韵文散文造诣皆比较杰出的大家。苏轼的散文为唐宋四家（韩愈、柳宗元、欧苏）之末，与唐代的古文运动发起者韩愈并称为「韩潮苏海」，也与欧阳修并称「欧苏」；更与父亲苏洵、弟苏辙合称「三苏」，父子三人，同列唐宋八大家。苏轼之诗与黄庭坚并称「苏黄」，又与陆游并称「苏陆」；其词「以诗入词」，首开词坛「豪放」一派，振作了晚唐、五代以来绮靡的西昆体余风。后世与南宋辛弃疾并称「苏辛」，惟苏轼故作豪放，其实清朗；其赋亦颇有名气，最知名者为贬谪期间借题发挥写的前后《赤壁赋》。宋代每逢科考常出现其文命题之考试，故当时学者曰：「苏文熟，吃羊肉、苏文生，嚼菜羹」。艺术方面，书法名列「苏、黄、米、蔡」北宋四大书法家（宋四家）之首；其画则开创了湖州画派；并在题画文学史上占有举足轻重的地位。'

doc = { 'DID': 'D001',
    'DTEXT': '苏轼（1037年1月8日－1101年8月24日），眉州眉山（今四川省眉山市）人，北宋时著名的文学家、政治家、艺术家、医学家。字子瞻，一字和仲，号东坡居士、铁冠道人。嘉佑二年进士，累官至端明殿学士兼翰林学士，礼部尚书。南宋理学方炽时，加赐谥号文忠，复追赠太师。有《东坡先生大全集》及《东坡乐府》词集传世，宋人王宗稷收其作品，编有《苏文忠公全集》。其散文、诗、词、赋均有成就，且善书法和绘画，是文学艺术史上的通才，也是公认韵文散文造诣皆比较杰出的大家。苏轼的散文为唐宋四家（韩愈、柳宗元、欧苏）之末，与唐代的古文运动发起者韩愈并称为「韩潮苏海」，也与欧阳修并称「欧苏」；更与父亲苏洵、弟苏辙合称「三苏」，父子三人，同列唐宋八大家。苏轼之诗与黄庭坚并称「苏黄」，又与陆游并称「苏陆」；其词「以诗入词」，首开词坛「豪放」一派，振作了晚唐、五代以来绮靡的西昆体余风。后世与南宋辛弃疾并称「苏辛」，惟苏轼故作豪放，其实清朗；其赋亦颇有名气，最知名者为贬谪期间借题发挥写的前后《赤壁赋》。宋代每逢科考常出现其文命题之考试，故当时学者曰：「苏文熟，吃羊肉、苏文生，嚼菜羹」。艺术方面，书法名列「苏、黄、米、蔡」北宋四大书法家（宋四家）之首；其画则开创了湖州画派；并在题画文学史上占有举足轻重的地位。',
    'QUESTIONS': [
        {'QID': 'D001Q01', 'QTEXT': '苏东坡的爸爸叫什么名字?', 'QTYPE': '基礎題', 'AMODE': ['Single-Span-Extraction'], 'ATYPE': 'Person'},
    ],
}

with open('data/raw/1.7.13/FGC_release_all_dev.json', encoding='utf-8') as f:
	docs = json.load(f)

class WikiQATest(TestCase):
    def test_predict_on_q(self):
        wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                        wikidata_ip=MONGODB_IP,
                        pred_infer='rule',
                        mode='prod',
                        check_atype_w_NE_type=False,
                        match_in_passage=False,
                        use_NE_span_in_psg=False,
                        use_se='None',
                        output_longest_answer=False,
                        verbose=True)

        ans = wiki_qa.predict_on_q(q)
        self.assertEqual(set(ans), set(['苏洵', '苏明允', '苏老泉', '明允', '老泉']))

    def test_predict_on_p_q(self):
        wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                        wikidata_ip=MONGODB_IP,
                        pred_infer='rule',
                        mode='prod',
                        check_atype_w_NE_type=False,
                        match_in_passage=True,
                        use_NE_span_in_psg=True,
                        use_se='None',
                        output_longest_answer=True,
                        verbose=True)
        ans = wiki_qa.predict_on_q_doc(q, p)
        self.assertEqual(ans, '苏洵')

    def test_predict_on_qs_of_one_doc(self):
        wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                        wikidata_ip=MONGODB_IP,
                        pred_infer='rule',
                        check_atype_w_NE_type=False,
                        match_in_passage=True,
                        use_NE_span_in_psg=True,
                        output_longest_answer=True,
                        mode='prod',
                        use_se='None',
                        verbose=True)

        ans = wiki_qa.predict_on_qs_of_one_doc(doc)
        self.assertEqual(ans, [{'AMODULE': 'Wiki-Kb-Inference',
                                'ATEXT': '苏洵',
                                'end_score': 0.0,
                                'score': 1.0,
                                'start_score': 0.0}])

    def test_on_dataset(self):
        wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                        wikidata_ip=MONGODB_IP,
                        pred_infer='rule',
                        mode='prod',
                        verbose=False)
        not_empty_answers = []
        for doc in docs:
            output = wiki_qa.predict_on_qs_of_one_doc(doc)
            if output[0]['ATEXT']:
                not_empty_answers.extend(output)
        self.assertGreater(len(not_empty_answers), 0)
        pprint(not_empty_answers)

    def test_predict_neural(self):
        wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                    wikidata_ip=MONGODB_IP,
                    pred_infer='neural',
					neural_model_fpath = 'files/v2_with_nones_early_stop_run2',
					dataset_fpath = 'files/predicate_inference_combined_v2_with_nones.csv',
					tokenizer_fpath = 'files/bert-base-chinese-vocab.txt',
                    mode='prod')
        not_empty_answers = []
        for doc in docs:
            output = wiki_qa.predict_on_qs_of_one_doc(doc)
            if output[0]['ATEXT']:
                not_empty_answers.extend(output)
        self.assertGreater(len(not_empty_answers), 0)
        pprint(not_empty_answers)

if __name__ == '__main__':
    import unittest
    unittest.main()
