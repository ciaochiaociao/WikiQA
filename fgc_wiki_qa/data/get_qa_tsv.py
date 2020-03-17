#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from ..utils.fgc_utils import data_to_csv
from ..utils.utils import load_json
import click

FGC_PATH = '../../data/processed/FGC_release_all(cn)_filtered2.json'


@click.command()
@click.argument('input_fpath')
@click.argument('output_fpath')
def main(input_fpath=FGC_PATH, output_fpath='fgc_qa_filtered.tsv'):
    d = load_json(input_fpath)
    with open(output_fpath, 'w', encoding='utf-8') as f:
        data_to_csv(d, f)


if __name__ == '__main__':
    main()