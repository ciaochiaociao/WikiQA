#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from unittest import TestCase

from neural_predicate_inference.predict import predicate_inference
from predicate_inference_neural import parse_question_w_neural


class Test(TestCase):
    def test_predicate_inference(self):
        self.assertEqual(predicate_inference('[ENTITY]的主席叫什麼名字?'), 'P488')

    def test_parse_question_w_neural(self):
        self.assertEqual(parse_question_w_neural('[ENTITY]的主席叫什麼名字?'),
                         ('[ENTITY]的主席叫什麼名字?', '領導者', (0, 17)))
