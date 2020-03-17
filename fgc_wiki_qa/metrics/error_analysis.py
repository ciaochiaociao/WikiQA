#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from pandas import read_csv, merge


def error_analysis(fgc_qa_fpath, fgc_wiki_benchmark_fpath, file4eval_fpath, output_fpath):
    df_fgc = read_csv(fgc_qa_fpath, header=0, sep='\t')
    df_results = read_csv(file4eval_fpath, header=0, sep='\t')
    df_fgc_wiki_benchmark = read_csv(fgc_wiki_benchmark_fpath, header=0, sep='\t')
    df1 = df_results[(df_results.answer.notna()) & (df_results.answer != 'None')]
    merged1 = merge(df1, right=df_fgc, how='left', on='qid')
    merged2 = merge(merged1, right=df_fgc_wiki_benchmark, how='left', left_on='qid', right_on='id')
    merged3 = merged2.reindex(columns=['qid', 'qtext', 'spo', 'parsed_subj', 'parsed_pred', 'pid', 'sid_x', 'sid_y',
                                       'pretty_values', 'oid', 'proc_values', 'answers', 'answer', 'atext'])
    merged3.to_excel(output_fpath, sheet_name='error_analysis')


def main():
    file4eval_fpath = 'experiments/file4eval_filtered.tsv'
    fgc_wiki_benchmark_fpath = 'experiments/fgc_wiki_benchmark_v0.1.tsv'
    fgc_qa_fpath = 'experiments/fgc_qa_filtered.tsv'
    output_fpath = 'error_analysis.xlsx'
    error_analysis(fgc_qa_fpath, fgc_wiki_benchmark_fpath, file4eval_fpath, output_fpath)


if __name__ == '__main__':
    main()