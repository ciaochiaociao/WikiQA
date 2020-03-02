#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from fgc_utils import data_to_csv
from utils import load_json

fgc_path = 'FGC_release_all(cn)_filtered2.json'
d = load_json(fgc_path)
with open('fgc_qa_filtered.tsv', 'w', encoding='utf-8') as f:
    data_to_csv(d, f)
