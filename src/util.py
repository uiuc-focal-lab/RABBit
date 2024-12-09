import torch
import itertools

def convert_tensor(tuple_list):
    t = torch.tensor(tuple_list)
    return t.T.reshape(-1)


def max_cross_verified(cross_verified):
    if len(cross_verified) <= 0:
        return 0.0
    
    def powerset(iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    
    def intersect(x, y):
        for e in x:
            if e in y:
                return True
        return False
    verified_count = 0.0
    powerset_cross_verified = list(powerset(cross_verified))
    for i in range(len(powerset_cross_verified)):
        curr_set = []
        does_intersect = False
        for x in powerset_cross_verified[i]:
            if len(x) <= 0:
                does_intersect = True
                break
            if intersect(x, curr_set):
                does_intersect = True
                break
            curr_set += list(x)
        # import pdb; pdb.set_trace()
        if does_intersect is False:
            verified_count = max(verified_count, len(powerset_cross_verified[i]))
    return verified_count

def generate_indices(indices, threshold, count, use_entries = True):
    entries = threshold // count
    tuple_list = []
    if entries <= 0:
        return None
    if 2 <= count < threshold:
        tuple_list = []
        for combo in itertools.combinations(indices, count):
            tuple_list.append(combo)
            if use_entries and len(tuple_list) >= entries:
                return convert_tensor(tuple_list=tuple_list), tuple_list
        
    else:
        raise ValueError(f"We don't support cross executions of {count}")
    
    if len(tuple_list) > 0:
        return convert_tensor(tuple_list=tuple_list), tuple_list
    else:
        return None 


def max_disjoint_tuples(tuples, scores, k):
    n = len(tuples)
    selected_tuples = []
    max_score = 0
    memo = {}

    def backtrack(index, current_score, current_selection):
        nonlocal max_score, selected_tuples


        if (index, tuple(current_selection)) in memo:
            return memo[(index, tuple(current_selection))]


        if len(current_selection) == k or index == n:
            if current_score > max_score:
                max_score = current_score
                selected_tuples = current_selection[:]
            return current_score


        disjoint = True
        for selected_tuple in current_selection:
            if any(x in selected_tuple for x in tuples[index]):
                disjoint = False
                break

        new_score = current_score
        if disjoint:
            new_score = backtrack(index + 1, current_score + scores[index], current_selection + [tuples[index]])

        new_score = max(new_score, backtrack(index + 1, current_score, current_selection))

        memo[(index, tuple(current_selection))] = new_score
        return new_score

    backtrack(0, 0, [])

    return selected_tuples


def greedy_disjoint_tuples(tuples, scores, k):
    tuples_with_scores = list(zip(tuples, scores))
    sorted_tuples = sorted(tuples_with_scores, key=lambda x:  x[1], reverse=True)
    
    selected_tuples = []
    elements_covered = set()

    for tup, score in sorted_tuples:
        if not any(elem in elements_covered for elem in tup):
            selected_tuples.append((tup, score))
            elements_covered.update(tup)

    return [selected_tuple[0] for selected_tuple in selected_tuples][:k]

def top_tuple(tuples, scores):
    if not tuples or not scores:
        return None
    max_score = float('-inf')
    best_tuple = None
    for tup, score in zip(tuples, scores):
        if score > max_score:
            max_score = score
            best_tuple = tup
    
    return [best_tuple]

def timeout_handler(signum, frame):
    print("Function timed out. Continuing with the code...")
    raise TimeoutError("Function execution timed out")