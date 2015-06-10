import logging
import random
import numpy as np

log = logging.info if True else print


class LineGraph():
    def __init__(self, wts):
        """

        :param wts: A list of edge wts to initialize the graph. Needs to
        implement
            combine: A function that takes in two wts and tells if they
        should be combined or not. Also returns the combined weight/edge.
            strength: An integer strength of path
        """
        self.lchildren = []
        for i, wt in enumerate(wts):
            self.lchildren.append([[i+1, wt]])

        self.lchildren.append([])
        self.processed = []
        self.checked_gcs = []
        self.path_strength_till = {}
        self.find_parents()

    @property
    def last_node(self):
        return len(self.lchildren) - 1

    def find_parents(self):
        self.lparents = [[] for _ in self.lchildren]
        for parent, children in enumerate(self.lchildren):
            for child, wt in children:
                self.lparents[child].append([parent, wt])

    def process_node(self, idx):
        if idx in self.processed:
            log("Already did", idx)
            return

        log("In", idx)
        ichild = 0
        while ichild < len(self.lchildren[idx]):
            chld_id, chld_wt = self.lchildren[idx][ichild]
            self.process_node(chld_id)
            log("Back in", idx)

            igrandchild = 0
            while igrandchild < len(self.lchildren[chld_id]):
                gc_id, gc_wt = self.lchildren[chld_id][igrandchild]
                if (idx, gc_id) in self.checked_gcs:
                    log("Already checked {} ({}) {}".format(idx, chld_id, gc_id))
                    igrandchild += 1
                    continue

                do_combine, new_wt = chld_wt.combine(gc_wt)
                log("Checking {} {} {} {}".format(idx, chld_id, gc_id,
                                                  do_combine))
                self.checked_gcs.append((idx, gc_id))

                if do_combine:
                    self.lchildren[idx].append([gc_id, new_wt])
                    log("Added {} to {}: {}".format(gc_id, idx,
                                                    self.lchildren[idx]))

                igrandchild += 1

            ichild += 1

        log("Done with ", idx)
        self.processed.append(idx)

    def process_tree(self):
        self.process_node(0)
        self.find_parents()

    def get_paths(self, n=0):
        if len(self.lchildren[n]) == 0:
            yield [n]

        for child, wt in self.lchildren[n]:
            for sub_path in self.get_paths(child):
                yield [n] + sub_path

    def path_strength(self, path):
        ret = 0
        for i in range(len(path)-1):
            for child, wt in self.lchildren[path[i]]:
                if child == path[i+1]:
                    ret += wt.strength()
                    break
            else:
                raise ValueError("Path not found in graph: {}".format(path))

        return ret

    def strongest_path(self, node=None):
        if node is None:
            node = self.last_node

        log("IN {}".format(node))
        if node in self.path_strength_till:
            log("Already did {}".format(node))

        elif len(self.lparents[node]) == 0:
            self.path_strength_till[node] = 0, [node]
            log("At root {}".format(node))

        else:
            best_strength, best_path = -np.inf, []
            for parent, wt in self.lparents[node]:
                strength, path_till = self.strongest_path(parent)
                log("Checked parent {} of {}".format(parent, node))
                log("\tGot: {} {}(+{})".format(strength, path_till,
                                               wt.strength()))
                strength += wt.strength()
                if strength > best_strength:
                    best_strength = strength
                    best_path = path_till

            self.path_strength_till[node] = best_strength, best_path + [node]

        return self.path_strength_till[node]

    def __str__(self):
        ret = ""
        for parent, children in enumerate(self.lchildren):
            ret += "\nParent {}: ".format(parent)
            for child in children:
                ret += "{}, ".format(child)

        ret += "\n"
        for child, parents in enumerate(self.lparents):
            ret += "\nChild {}: ".format(child)
            for parent in parents:
                ret += "{}, ".format(parent)

        return ret


class Weight():
    def __init__(self, val):
        self.val = val

    def combine(self, other):
        return random.random() < .3, Weight(int(100*random.random())+(
            self.val+other.val)//2)

    def strength(self):
        return self.val

    def __repr__(self):
        return "{}".format(self.val)


if __name__ == "__main__":
    weights = [Weight(val) for val in range(10, 80, 10)]
    print(list(enumerate(weights)))
    lt = LineGraph(weights)
    print(lt.lchildren)
    print(lt)
    lt.process_tree()
    print(lt)

    paths = lt.get_paths()
    for path in paths:
        print(path, lt.path_strength(path))
    print("Strongest Path: ", lt.strongest_path())