import re
from enum import Enum

RESULT_DIR = 'results/'


def strip_name(obj, pos=-1):
    return re.split('\.|/', str(obj))[pos]


# Domains used for verification
class Domain(Enum):
    DEEPZ = 1
    DEEPPOLY = 2
    BOX = 3
    LP = 4
    LIRPA_IBP = 5
    LIRPA_CROWN = 6
    LIRPA_CROWN_IBP = 7
    LIRPA_CROWN_OPT = 8
    LIRPA_CROWN_FORWARD = 9
    UAP_BASIC = 10
    UAP_BASIC_LP = 11
    UAP_DIFF = 12


# Used for status of the complete verifier
class Status(Enum):
    VERIFIED = 1
    ADV_EXAMPLE = 2
    UNKNOWN = 3
    MISS_CLASSIFIED = 4
