import click
import pandas as pd
import numpy as np

@click.command()
@click.argument('dataset_fpath', default='../data/external/manual_after_auto_dataset.xlsx')
@click.argument('result_fpath', default='../data/external/fgc_predicate_inference_v0.1.tsv')
def main(dataset_fpath, result_fpath):
    df = pd.read_excel(dataset_fpath)

    def get_label(df):
        return np.where((df['spo'] == 'sp?') &
                        (df['plabel'].notna()),
                        df['plabel'],
                        'None')

    df = df.assign(label=get_label)

    df = df[['qid', 'qtext', 'label']]

    df.to_csv(result_fpath, index=None)


if __name__ == '__main__':
    main()

