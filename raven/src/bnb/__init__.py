from enum import Enum


class Split(Enum):
    NONE = 0
    RELU_RAND = 1
    RELU_GRAD = 2
    RELU_ESIP_SCORE = 3
    INPUT = 4
    INPUT_GRAD = 5
    # Smart Branching (SB)
    INPUT_SB = 6


def is_relu_split(split):
    relu_splits = [Split.RELU_RAND, Split.RELU_ESIP_SCORE, Split.RELU_GRAD]
    if split in relu_splits:
        return True
    return False


def is_input_split(split):
    input_splits = [Split.INPUT, Split.INPUT_GRAD, Split.INPUT_SB]
    if split in input_splits:
        return True
    return False
