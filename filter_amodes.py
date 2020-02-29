#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from fgc_utils import filter_out_amodes_and_save


amodes = ['Yes-No', 'Comparing-Members', 'Kinship', 'Arithmetic-Operations',  'Multi-Spans-Extraction', 'Counting']
filter_out_amodes_and_save('FGC_release_all(cn).json', amodes, 'FGC_release_all(cn)_filtered2.json')