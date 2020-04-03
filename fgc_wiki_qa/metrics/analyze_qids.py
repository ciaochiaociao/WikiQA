import json
import sys

import click

from fgc_wiki_qa.utils.utils import TeeLogger


def get_stubborn_errors(already_errors, errors):
    return list(set(already_errors) & set(errors))


def get_robust_corrects(already_corrects, corrects):
    return list(set(already_corrects) & set(corrects))


def get_betrayed_corrects(already_corrects, errors):
    return list(set(already_corrects) & set(errors))


def get_corrected_errors(already_errors, corrects):
    return list(set(already_errors) & set(corrects))


def get_gained_corrects(already_corrects, corrects, already_errors):
    return list(set(corrects) - set(already_corrects) - set(get_corrected_errors(already_errors, corrects)))


def get_sacrificed_corrects(already_corrects, corrects, errors):
    return list((set(already_corrects) - set(corrects)) - set(get_betrayed_corrects(already_corrects, errors)))


def get_new_errors(already_errors, errors, already_corrects):
    return list((set(errors) - set(already_errors)) - set(get_betrayed_corrects(already_corrects, errors)))


def get_eliminated_errors(already_errors, errors, corrects):
    return list((set(already_errors) - set(errors)) - set(get_corrected_errors(already_errors, corrects)))


def get_tp_predicates(parsed, golds):
    return list(set(parsed) & set(golds))


def get_fn_predicates(parsed, golds):
    return list(set(golds) - set(parsed))


def get_fp_predicates(parsed, golds):
    return list(set(parsed) - set(golds))


@click.command()
@click.option('--qids_fpath')
@click.option('--already_qids_fpath')
@click.option('--save_fpath')
@click.option('--report_fpath')
def analyze(qids_fpath, already_qids_fpath, save_fpath, report_fpath) -> dict:

    sys.stdout = TeeLogger(report_fpath)

    with open(qids_fpath) as f:
        qids = json.load(f)
        corrects = qids['corrects']
        errors = qids['errors']

    with open(already_qids_fpath) as f:
        already_qids = json.load(f)
        already_corrects = already_qids['corrects']
        already_errors = already_qids['errors']

    qids_dict = {
        'corrects': corrects,
        'errors': errors,
        'already_corrects': already_corrects,
        'already_errors': already_errors,
        'sacrificed_corrects': get_sacrificed_corrects(already_corrects, corrects, errors),
        'gained_correts': get_gained_corrects(already_corrects, corrects, already_errors),
        'new_errors': get_new_errors(already_errors, errors, already_corrects),
        'eliminated_errors': get_eliminated_errors(already_errors, errors, corrects),
        'stubborn_errors': get_stubborn_errors(already_errors, errors),
        'robust_corrects': get_robust_corrects(already_corrects, corrects),
        'betrayed_corrects': get_betrayed_corrects(already_corrects, errors),
        'corrected_errors': get_corrected_errors(already_errors, corrects),
    }
    print(f'Analyzing and saving to {save_fpath} ...')
    print('[INFO] Corrects: ', qids_dict['corrects'])
    print('[INFO] Errors: ', qids_dict['errors'])
    if qids_dict['robust_corrects']:
        print('[INFO] Robust Corrects (C>C): ', qids_dict['robust_corrects'])
    if qids_dict['stubborn_errors']:
        print('[INFO] Stubborn Errors (E>E): ', qids_dict['stubborn_errors'])
    if qids_dict['betrayed_corrects']:
        print('[BAD] Betrayed Corrects (C>E): ', qids_dict['betrayed_corrects'])
    if qids_dict['sacrificed_corrects']:
        print('[BAD] Sacrificed (C>_): ', qids_dict['sacrificed_corrects'])
    if qids_dict['new_errors']:
        print('[BAD] New Errors (_>E): ', qids_dict['new_errors'])
    if qids_dict['corrected_errors']:
        print('[GOOD] Corrected Errors (E>C): ', qids_dict['corrected_errors'])
    if qids_dict['eliminated_errors']:
        print('[GOOD] Eliminated Errors (E>_): ', qids_dict['eliminated_errors'])
    if qids_dict['gained_correts']:
        print('[GOOD] Gained (_>C): ', qids_dict['gained_correts'])

    with open(save_fpath, 'w', encoding='utf-8') as f:
        json.dump(qids_dict, f, ensure_ascii=False, indent=4)

    return qids_dict


if __name__ == '__main__':
    analyze()