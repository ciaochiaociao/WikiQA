#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

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