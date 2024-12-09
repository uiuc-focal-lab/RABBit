
import heapq
import torch

class TreeNode:
    def __init__(self, children = [], As = [], logit_lbs = [], uap_accuracy = 0.0, time = None,
                cross_ex_loss = None, targetted_loss = None, 
                topk_loss=None):
        self.children = children
        self.As = As
        self.logit_lbs = logit_lbs
        self.uap_accuracy = uap_accuracy
        self.cross_ex_loss = cross_ex_loss
        self.targetted_loss = targetted_loss
        self.topk_loss = topk_loss
        self.time = time
    
    def add_child(self, child):
        self.children.append(child)
        self.children[-1].children = []
    

class PQNode:
    def __init__(self, net = None, domains = [], input_idx = None, tree_node = None):
        self.net = net
        self.domains = domains
        self.input_idx = input_idx
        self.tree_node = tree_node

class SpecificationTree:
    def __init__(self, root = None):
        self.root = root 
        self.root.children = []
    
    def get_tree_leaves(self):
        if not self.root:
            return []

        stack = [self.root]
        leaves = []

        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)

        return leaves
    
    def get_res(self):
        if not self.root:
            return 0.0, [], [], []

        stack = [self.root]
        
        As = []
        logit_lbs = []
        
        num_images = self.root.logit_lbs.shape[0]
        is_verified = torch.zeros(num_images, dtype = bool)
        
        while stack:
            node = stack.pop()
            if not node.children:
                As.append(node.As)
                logit_lbs.append(node.logit_lbs)
                min_vals = node.logit_lbs.min(dim = 1).values.cpu()
                try:
                    is_verified += (min_vals < 0) + (min_vals.isnan())
                except:
                    import pdb;pdb.set_trace()
                    raise RuntimeError
            else:
                stack.extend(node.children)
        is_verified = (~is_verified).tolist()
        return sum(is_verified)/num_images, is_verified, logit_lbs, As
    
    def get_cross_ex_res(self):
        if not self.root:
            return []

        stack = [self.root]
        cross_ex_losses = []

        while stack:
            node = stack.pop()
            if not node.children:
                cross_ex_losses.append(node.cross_ex_loss)
                
            else:
                stack.extend(node.children)
        
        return cross_ex_losses
    
    
    def get_lbs_res(self):
        if not self.root:
            return []

        stack = [self.root]
        cross_ex_losses = []

        while stack:
            node = stack.pop()
            if not node.children:
                cross_ex_losses.append(node.logit_lbs)
                
            else:
                stack.extend(node.children)
        
        return cross_ex_losses
    
    def get_targetted_res(self):
        if not self.root:
            return []

        stack = [self.root]
        
        targetted_losses = []

        while stack:
            node = stack.pop()
            if not node.children:
                targetted_losses.append(node.targetted_loss)
                
            else:
                stack.extend(node.children)
        
        return targetted_losses

    def get_topk_loss(self):
        if not self.root:
            return []

        stack = [self.root]
        
        topk_losses = []

        while stack:
            node = stack.pop()
            if not node.children:
                topk_losses.append(node.topk_loss)  
            else:
                stack.extend(node.children)
        
        return topk_losses

    def get_time_res(self):
        if not self.root:
            return []

        stack = [self.root]
        
        targetted_losses = []

        while stack:
            node = stack.pop()
            if not node.children:
                targetted_losses.append(node.time)
                
            else:
                stack.extend(node.children)
        
        return targetted_losses

    
    def num_group_verified(self):
        if not self.root:
            return 0
        
        stack = [self.root]
        
        
        not_verified = torch.zeros(self.root.cross_ex_loss.shape[0], dtype = bool)
        
        while stack:
            node = stack.pop()
            if not node.children:
                not_verified += (node.cross_ex_loss < 0).squeeze()
                
            else:
                stack.extend(node.children)
        
        return sum(~not_verified)
        

    
    def __repr__(self):
        if self.root is None:
            return "Empty Tree"
        elif len(self.root.logit_lbs) == 1:
            return self._tree_repr_single(self.root)

        return self._tree_repr_crossex(self.root)

    def _tree_repr_single(self, node, level=0):
        tree_repr = ""
        if node is None:
            return tree_repr
        if level > 0:
            tree_repr += "  " * (level - 1) + " └─╴" + str(node.uap_accuracy) + "\n"
        else:
            tree_repr += str(node.uap_accuracy) + "\n"
        for child in node.children:
            tree_repr += self._tree_repr_single(child, level + 1)
        return tree_repr
    
    def _tree_repr_crossex(self, node, level=0):
        tree_repr = ""
        if node is None:
            return tree_repr
        if level > 0:
            tree_repr += "  " * (level - 1) + " └─╴" + str(node.cross_ex_loss.tolist()) + "\n"
        else:
            tree_repr += str(node.cross_ex_loss.tolist()) + "\n"
        for child in node.children:
            tree_repr += self._tree_repr_crossex(child, level + 1)
        return tree_repr
    
    def tree_height(self):
        if not self.root:
            return 0

        queue = [self.root]
        height = 0

        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.pop(0)
                for child in node.children:
                    queue.append(child)
            height += 1
        return height

class ProofPQ:
    def __init__(self, key=lambda x: x):
        self._queue = []
        self.key = key

    def push(self, item):
        heapq.heappush(self._queue, (self.key(item), id(item), item))

    def pop(self, n = 1):
        return [heapq.heappop(self._queue)[-1] for _ in range(n)]
    
    def __len__(self):
        return len(self._queue)
