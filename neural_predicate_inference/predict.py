#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import os

import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from collections import Counter
#
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

wd = os.path.dirname(__file__)

datasat_fpath = os.path.join(wd, 'models/predicate_inference_combined_v2_with_nones.csv')
model_fpath = os.path.join(wd, 'models/v2_with_nones_early_stop.h5')
tokenizer_fpath = os.path.join(wd, 'models/bert-base-chinese-vocab.txt')

df = pd.read_csv(datasat_fpath, header=0, encoding='utf-8')
all_predicates = Counter(df.pid).most_common()
all_predicates = [p[0] for p in all_predicates]
# all_predicates = ['None', 'P800', 'P488', 'P166', 'P50', 'alias', 'P1787', 'P571', 'P27', 'P279', 'P161', 'P57', 'P19', 'P36', 'P1128', 'P2048', 'P112', 'P159', 'P569', 'P856', 'P169', 'P1056', 'P414', 'P2226', 'P127', 'P17', 'P2295', 'P2139', 'P740', 'P7729', 'P355', 'P452', 'P2541']


print('loading tokenizer ...')
tokenizer = BertTokenizer(tokenizer_fpath)

special_tokens_dict = {'additional_special_tokens': ['[ENTITY]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


def encode_template(template):
    return tf.constant(tokenizer.encode(template, add_special_tokens=True))[None, :]


def predict(template):
    return model.predict(encode_template(template))[0]


def predicate_inference(template):
    return all_predicates[tf.argmax(predict(template)).numpy()]


BATCH_SIZE = 64

print('creating model')
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese',
                                                        num_labels=len(all_predicates))

print('loading pretrained model ...')
model.load_weights(model_fpath)

print('test ... ', '(input) [ENTITY]的主席叫什麼名字? (output)', predicate_inference('[ENTITY]的主席叫什麼名字?'))

