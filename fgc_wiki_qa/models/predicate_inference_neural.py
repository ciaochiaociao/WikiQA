#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from .neural_predicate_inference.predict import predicate_inference
from ..utils.wikidata4fgc_v2 import get_all_aliases_from_pid


def parse_question_w_neural(qtext):
    """parse question with neural and rules

    :param qtext: question/template text
    :return Union[(str, str, Tuple[int]), False]: subject, attr_name, span_of_all_possible_subjects
    """
    # neural predicate inference
    pid = predicate_inference(qtext)
    try:
        attr_name = get_all_aliases_from_pid(pid)[0][0]  # take the main label  # TODO: use PID instead
    except IndexError:
        return False

    # TODO: parse subject
    subj_span = (0, len(qtext))
    subj = qtext

    return subj, attr_name, subj_span


