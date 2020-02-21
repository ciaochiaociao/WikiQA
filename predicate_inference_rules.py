#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import re

from predicate_inference_regex import strict_label_map


def parse_question_by_regex(qtext):

    # TODO
    # for attr_name, alias_name in alias_map.items():
    #     attr_name, alias_name

    for attr, patterns in strict_label_map.items():
        for pattern in patterns:
            pattern = re.compile(pattern)
            result = pattern.search(qtext)
            if result and 'name' in result.groupdict():
                return result.groupdict()['name'], attr, result.span(1), pattern.pattern

    return False
