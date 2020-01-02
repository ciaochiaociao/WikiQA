import re
from ansi.color import fg


def in_ranges(ix, ranges):
    for tok_b, tok_e in ranges:
        if tok_b <= ix < tok_e:
            return True
    return False


def highlight(text, match, output=False):
    highlighted = re.sub(r'(' + match + ')', fg.red(r'\1'), text)
    if output:
        print(highlighted)
    else:
        return highlighted


def default_answer(did, answer_module, atext='', score=0.0):
    ans = []

    for qid in did['QUESTIONS']:
        ans.append(
            [
                default_acand(answer_module, atext, score)
            ]
        )

    return ans


def default_acand(answer_module, atext, score):
    return {
        'AMODULE': answer_module,
        'ATEXT': atext,
        'score': score
     }
