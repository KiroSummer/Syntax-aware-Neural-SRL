class Tree(object):
    def __init__(self, index):
        self.parent = None
        self.is_left = False
        self.index = index
        self.left_children = list()
        self.left_num = 0
        self.right_children = list()
        self.right_num = 0
        self._depth = -1
        self.order = []

    def add_left(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = True
        self.left_children.append(child)
        self.left_num += 1

    def add_right(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = False
        self.right_children.append(child)
        self.right_num += 1

    def size(self):  # compute the total size of the Tree
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.left_num):
            count += self.left_children[i].size()
        for i in range(self.right_num):
            count += self.right_children[i].size()
        self._size = count
        return self._size

    def depth(self):  # compute the depth of the Tree
        if self._depth > 0:
            return self._depth
        count = 0
        if self.left_num + self.right_num > 0:
            for i in range(self.left_num):
                child_depth = self.left_children[i].depth()
                if child_depth > count:
                    count = child_depth
            for i in range(self.right_num):
                child_depth = self.right_children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def traverse(self):  # traverse the Tree
        if len(self.order) > 0:
            return self.order

        for i in range(self.left_num):
            left_order = self.left_children[i].traverse()
            self.order.extend(left_order)
        for i in range(self.right_num):
            right_order = self.right_children[i].traverse()
            self.order.extend(right_order)
        self.order.append(self.index)  # append the root
        return self.order

    def traverse_to_root(self):
        if self.parent is None:
            return [self.index]
        order = []
        current_node = self
        while current_node is not None:
            order.append(current_node.index)
            current_node = current_node.parent
        return order


def creatTree(heads):
    tree = []
    # current sentence has already been numberized [form, head, rel]
    root = None
    for idx, head in enumerate(heads):
        tree.append(Tree(idx))

    for idx, head in enumerate(heads):
        if head == -1:  # -1 mszhang, 0 kiro
            root = tree[idx]
            continue
        if head < 0:
            print('error: multi roots')
        if head > idx:
            tree[head].add_left(tree[idx])
        if head < idx:
            tree[head].add_right(tree[idx])
        if head == idx:
            print('error: head is it self.')
       
    return root, tree


def find_sub_paths(argument_path, predicate_path):
    is_find = False
    common_ancestor_in_predicate_path = -1
    for i, a in enumerate(argument_path):
        for j, p in enumerate(predicate_path):
            if a == p:
                is_find = True
                common_ancestor_in_predicate_path = j
                break
        if is_find is True:
            assert common_ancestor_in_predicate_path != -1
            return argument_path[:i + 1], predicate_path[:common_ancestor_in_predicate_path + 1]
    print("There is something wrong is finding the common ancestor!")
    exit()


def find_sentence_sub_paths(trees, predicate):
    predicate = int(predicate)
    word_to_root_paths = []
    for tree in trees:
        root_path = tree.traverse_to_root()
        word_to_root_paths.append(root_path)
    predicate_root_path = word_to_root_paths[predicate]
    sentence_sub_paths = []
    for i, word_path in enumerate(word_to_root_paths):  # does the left path or the right path matters?
        if i <= predicate:
            a_path, p_path = find_sub_paths(word_path, predicate_root_path)
            sentence_sub_paths.append([a_path, p_path])
        else:
            p_path, a_path = find_sub_paths(predicate_root_path, word_path)
            sentence_sub_paths.append([p_path, a_path])

    return sentence_sub_paths
