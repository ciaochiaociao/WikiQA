from pprint import pprint
import json
from fgc_wiki_qa.models.wikiqa import WikiQA

doc = { 'DID': 'D001',
    'DTEXT': '苏轼（1037年1月8日－1101年8月24日），眉州眉山（今四川省眉山市）人，北宋时著名的文学家、政治家、艺术家、医学家。字子瞻，一字和仲，号东坡居士、铁冠道人。嘉佑二年进士，累官至端明殿学士兼翰林学士，礼部尚书。南宋理学方炽时，加赐谥号文忠，复追赠太师。有《东坡先生大全集》及《东坡乐府》词集传世，宋人王宗稷收其作品，编有《苏文忠公全集》。其散文、诗、词、赋均有成就，且善书法和绘画，是文学艺术史上的通才，也是公认韵文散文造诣皆比较杰出的大家。苏轼的散文为唐宋四家（韩愈、柳宗元、欧苏）之末，与唐代的古文运动发起者韩愈并称为「韩潮苏海」，也与欧阳修并称「欧苏」；更与父亲苏洵、弟苏辙合称「三苏」，父子三人，同列唐宋八大家。苏轼之诗与黄庭坚并称「苏黄」，又与陆游并称「苏陆」；其词「以诗入词」，首开词坛「豪放」一派，振作了晚唐、五代以来绮靡的西昆体余风。后世与南宋辛弃疾并称「苏辛」，惟苏轼故作豪放，其实清朗；其赋亦颇有名气，最知名者为贬谪期间借题发挥写的前后《赤壁赋》。宋代每逢科考常出现其文命题之考试，故当时学者曰：「苏文熟，吃羊肉、苏文生，嚼菜羹」。艺术方面，书法名列「苏、黄、米、蔡」北宋四大书法家（宋四家）之首；其画则开创了湖州画派；并在题画文学史上占有举足轻重的地位。',
    'QUESTIONS': [
        {'QID': 'D001Q01', 'QTEXT': '苏东坡的爸爸叫什么名字?', 'QTYPE': '基礎題', 'AMODE': ['Single-Span-Extraction'], 'ATYPE': 'Person'},
    ],
}

CORENLP_IP = 'http://140.109.19.191:9000'
MONGODB_IP = 'mongodb://140.109.19.51:27020'

wiki_qa = WikiQA(corenlp_ip=CORENLP_IP,
                wikidata_ip=MONGODB_IP,
                pred_infer='rule',
                mode='prod',
                use_se='None',
                verbose=True)
all_answers = wiki_qa.predict_on_qs_of_one_doc(doc)
pprint(all_answers)
