#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from opencc import OpenCC

cc = OpenCC('t2s')
DEFAULT_CORENLP_IP = 'http://140.109.19.191:9000'
FGC_KB_PATH = 'fgc_knowledgebase.json'
UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=True)'