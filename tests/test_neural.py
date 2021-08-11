#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from unittest import TestCase

from fgc_wiki_qa.models.neural_predicate_inference.predict import NeuralPredicateInferencer
from fgc_wiki_qa.models.predicate_inference_neural import parse_question_w_neural

npi = NeuralPredicateInferencer(
    model_fpath = 'files/v2_with_nones_early_stop_run2',
    dataset_fpath = 'files/predicate_inference_combined_v2_with_nones.csv',
    tokenizer_fpath = 'files/bert-base-chinese-vocab.txt',
)

class NeuralTest(TestCase):

    def test_predicate_inference(self):
        self.assertEqual(npi.predicate_inference('[ENTITY]的主席叫什麼名字?'), 'P488')

    def test_parse_question_w_neural(self):
        self.assertEqual(parse_question_w_neural(npi, '[ENTITY]的主席叫什麼名字?'),
                         ('[ENTITY]的主席叫什麼名字?', '領導者', (0, 17)))

