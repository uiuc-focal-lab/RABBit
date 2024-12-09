import random

from raven.src import config
from src.bnb import Split
from src.common import Status
from src.proof_transfer.pt_types import ProofTransferMethod


class Reluspec:
    def __init__(self, relu_mask):
        self.relu_mask = relu_mask
        self.status = Status.UNKNOWN

    def split_spec(self, split, chosen_relu_id):
        relu_mask = self.relu_mask

        relu_mask1 = {}
        relu_mask2 = {}

        for relu in relu_mask.keys():
            if relu == chosen_relu_id:
                relu_mask1[relu] = -1
                relu_mask2[relu] = 1
            else:
                relu_mask1[relu] = relu_mask[relu]
                relu_mask2[relu] = relu_mask[relu]

        return [Reluspec(relu_mask1), Reluspec(relu_mask2)]

    def choose_relu(self, split_score=None, inp_template=None, args=None):
        """
        Chooses the relu that is split in branch and bound.
        @param: relu_spec contains relu_mask which is a map that maps relus to -1/0/1. 0 here indicates that the relu
            is ambiguous
        """
        relu_mask = self.relu_mask
        split = args.split

        if split == Split.RELU_RAND:
            all_relus = []

            # Collect all unsplit relus
            for relu in relu_mask.keys():
                if relu_mask[relu] == 0 and relu[0] == 2:
                    all_relus.append(relu)

            return random.choice(all_relus)

        elif split == Split.RELU_GRAD or split == Split.RELU_ESIP_SCORE:
            # Choose the ambiguous relu that has the maximum score in relu_score
            if split_score is None:
                raise ValueError("relu_score should be set while using relu_grad splitting mode")

            max_score = 0
            chosen_relu = None

            for relu in relu_mask.keys():
                if relu_mask[relu] == 0 and relu in split_score.keys():
                    if split_score[relu] >= max_score:
                        max_score = split_score[relu]
                        chosen_relu = relu

            if chosen_relu is None:
                raise ValueError("Attempt to split should only take place if there are ambiguous relus!")

            config.write_log("Chosen relu for splitting: " + str(chosen_relu) + " " + str(max_score))
            return chosen_relu
        else:
            # Returning just the first unsplit relu
            for relu in relu_mask.keys():
                if relu_mask[relu] == 0:
                    return relu
        raise ValueError("No relu chosen!")