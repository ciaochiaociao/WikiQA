#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

from opencc import OpenCC

cc = OpenCC('t2s')
DEFAULT_CORENLP_IP = 'http://140.109.19.191:9000'
FGC_KB_PATH = 'fgc_knowledgebase.json'
UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=True)'