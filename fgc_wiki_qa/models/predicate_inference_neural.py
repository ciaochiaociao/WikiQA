#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from ..utils.wikidata_utils import get_all_aliases_from_pid


def parse_question_w_neural(inferencer, qtext):
    """parse question with neural and rules

    :param qtext: question/template text
    :return Union[(str, str, Tuple[int]), False]: subject, attr_name, span_of_all_possible_subjects
    """
    # neural predicate inference
    pid = inferencer.predicate_inference(qtext)
    try:
        attr_name = get_all_aliases_from_pid(pid)[0][0]  # take the main label  # TODO: use PID instead
    except IndexError:
        return False

    # TODO: parse subject, here outputs the whole sentence as the subject to let the following query generator for EL linking to automatically create queries
    # e.g. The whole sentence will generate the queries of the spans which are NEs.
    subj_span = (0, len(qtext))
    subj = qtext

    return subj, attr_name, subj_span


