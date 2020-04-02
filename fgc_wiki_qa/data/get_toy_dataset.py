import json

import click

@click.command()
@click.argument('fgc_fpath')
@click.argument('output_fpath')
@click.option('--dids', multiple=True, default=["D071", "D097"])
def main(fgc_fpath, output_fpath, dids):

    with open(fgc_fpath, encoding='utf-8') as f:
        docs = json.load(f)
    results = []
    for doc in docs:
        if doc['DID'] in dids:
              results.append(doc)

    with open(output_fpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
