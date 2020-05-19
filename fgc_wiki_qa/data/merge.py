import json
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('jsons', nargs='*')
argparser.add_argument('out_fpath')

args = argparser.parse_args()

docs = []
for fpath in args.jsons:
    with open(fpath, encoding='utf-8') as f:
        docs.extend(json.load(f))

docs.sort(key=lambda d: d['QUESTIONS'][0]['QID'])

with open(args.out_fpath, 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False)
