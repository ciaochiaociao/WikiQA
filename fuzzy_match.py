#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import regex


def build_fuzzy_match_pattern(matcher):
    max_errors = 1
    if len(matcher) > 3:
        pattern = '(' + matcher + '){e<=' + str(max_errors) + '}'  # allowed max_errors
    else:
        pattern = matcher
    return pattern


def fuzzy_match(text, pattern):
    matches = list(regex.finditer(pattern, text))
    return matches