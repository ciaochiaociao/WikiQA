#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import json

from ..utils.fgc_utils import q_doc_generator, get_docs_with_certain_qs
from ..utils.utils import load_json
import click


@click.command()
@click.argument('fgc_fpath')
@click.argument('output_fpath')
def main(fgc_fpath='FGC_release_all(cn).json', output_fpath='FGC_release_all(cn)_filtered2.json'):
    # noamodes = ['YesNo', 'Comparing-Members', 'Kinship', 'Arithmetic-Operations', 'Multi-Spans-Extraction', 'Counting']
    amodes = ['Single-Span-Extraction', 'Date-Duration']
    noqtypes = ['申論']

    docs = load_json(fgc_fpath)
    g = q_doc_generator(docs)
    qids = []
    for q, doc in g:
        try:
            if set(q['AMODE']) & set(amodes) and q['QTYPE'] not in noqtypes:
                qids.append(q['QID'])
        except KeyError:  # 申论
            pass

    newdocs = get_docs_with_certain_qs(qids, docs)
    with open(output_fpath, 'w', encoding='utf-8') as f:
        json.dump(newdocs, f, ensure_ascii=False)


if __name__ == '__main__':
    main()