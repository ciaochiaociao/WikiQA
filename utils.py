import re
from ansi.colour import fg, bg

import json


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


def overlapped_range(ranges, overlapper):

    ixes, overlapped_ranges = [], []
    for ix, _range in enumerate(ranges):
        if set(range(*_range)).intersection(range(*overlapper)):
            ixes.append(ix); overlapped_ranges.append(_range)

    return ixes, overlapped_ranges


def apply_dict(being_applied, applying_dict):
    applied = tuple(applying_dict[i] for i in being_applied)
    return applied


def print_multicolor_strs(strs, strs_class, color_table=False,
                          classes_w_color=False, classes_wo_color=False, colors=False,
                          **kwargs):
    FG_COLORS = [fg.red, fg.green, fg.blue, fg.magenta, fg.cyan, fg.yellow]
    BG_COLORS = [bg.red, bg.green, bg.blue, bg.magenta, bg.cyan, bg.yellow]
    ALL_COLORS = FG_COLORS + BG_COLORS
    NONE_COLOR = fg.default

    all_classes = set(strs_class)
    if classes_wo_color and not classes_w_color:
        classes_w_color = all_classes - set(classes_wo_color)
    if classes_w_color and not classes_wo_color:
        classes_wo_color = all_classes - set(classes_w_color)
    if not classes_w_color and not classes_wo_color:
        classes_w_color, classes_wo_color = all_classes, {}

    num_classes_w_color = len(classes_w_color)
    num_classes_wo_color = len(classes_wo_color)
    assert num_classes_w_color + num_classes_wo_color == len(all_classes)

    if not color_table:

        # default classes order
        classes = list(classes_w_color) + list(classes_wo_color)  # classes w/o color should be at the end

        # default colors
        if not colors:
            # classes w/ color
            if num_classes_w_color <= 12:
                colors = ALL_COLORS[:num_classes_w_color]
            else:
                raise ValueError('# of classes with color cannot exceed 12')

            # classes wo color
            colors.extend([NONE_COLOR] * num_classes_wo_color)

        color_table = dict(zip(classes, colors))

    colors = [color_table[class_] for class_ in strs_class]
    colored_strs = [color(str_) for str_, color in zip(strs, colors)]
    print(' '.join(colored_strs), **kwargs)


def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data