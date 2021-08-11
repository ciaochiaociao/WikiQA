#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from collections import Counter
from typing import List

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

from ...config import DATASET_FPATH, TOKENIZER_FPATH, MODEL_FPATH
#
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class NeuralPredicateInferencer:
    def __init__(self, model_fpath=MODEL_FPATH, tokenizer_fpath=TOKENIZER_FPATH, dataset_fpath=DATASET_FPATH
                 , allow_growth=True, framework='tf2'):

        if framework == 'tf2':
            from transformers import TFBertForSequenceClassification as BertForSequenceClassification
            if allow_growth:
                # memory growth
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        # Currently, memory growth needs to be the same across GPUs
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    except RuntimeError as e:
                        # Memory growth must be set before GPUs have been initialized
                        print(e)
        elif framework == 'pt':
            from transformers import BertForSequenceClassification

        df = pd.read_csv(dataset_fpath, header=0, encoding='utf-8')
        all_predicates = Counter(df.pid).most_common()
        self.all_predicates = [p[0] for p in all_predicates]
        # all_predicates = ['None', 'P800', 'P488', 'P166', 'P50', 'alias', 'P1787', 'P571', 'P27', 'P279', 'P161', 'P57', 'P19', 'P36', 'P1128', 'P2048', 'P112', 'P159', 'P569', 'P856', 'P169', 'P1056', 'P414', 'P2226', 'P127', 'P17', 'P2295', 'P2139', 'P740', 'P7729', 'P355', 'P452', 'P2541']

        print('loading tokenizer ...')
        self.tokenizer = BertTokenizer(tokenizer_fpath)

        special_tokens_dict = {'additional_special_tokens': ['[ENTITY]']}
        self.num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print('creating model')

        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(self.all_predicates))

        self.model.load_weights(model_fpath)

    def encode_template(self, template, maxlen=512) -> List[List[int]]:

        return self.tokenizer.encode(template, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, return_tensors='tf')  # arg return_tensors automatically expands to 2D to form a one-example batch

    def predict(self, template):
        return self.model.predict(self.encode_template(template))[0]

    def predicate_inference(self, template):
        return self.all_predicates[tf.argmax(self.predict(template)).numpy()]


if __name__ == '__main__':

    npi = NeuralPredicateInferencer()
    print('test ... ', '(input) [ENTITY]的主席叫什麼名字? (output)', npi.predicate_inference('[ENTITY]的主席叫什麼名字?'))
    print('test ... ', '(input) [ENTITY]的主席叫什麼名? (output)', npi.predicate_inference('[ENTITY]的主席叫什麼名?'))