import numpy as np
import logging

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


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

        self.last_node = len(self.lchildren)
        self.lchildren.append([])

        self.processed = []
        self.checked_gcs = []
        self.path_strength_till = {}
        self.lparents = []
        self.find_parents()

    def find_parents(self):
        self.lparents = [[] for _ in self.lchildren]
        for parent, children in enumerate(self.lchildren):
            for child, wt in children:
                self.lparents[child].append([parent, wt])

    def process_node(self, idx):
        if idx in self.processed:
            logd("Already processed {}".format(idx))
            return

        logd("Processing in {}".format(idx))
        ichild = 0
        while ichild < len(self.lchildren[idx]):
            chld_id, chld_wt = self.lchildren[idx][ichild]
            self.process_node(chld_id)
            logd("Processing back in {}".format(idx))

            igrandchild = 0
            while igrandchild < len(self.lchildren[chld_id]):
                gc_id, gc_wt = self.lchildren[chld_id][igrandchild]
                if (idx, gc_id) in self.checked_gcs:
                    logd("Already checked {} ({}) {}".format(idx, chld_id, gc_id))
                    igrandchild += 1
                    continue

                do_combine, new_wt = chld_wt.combine(gc_wt)
                logd("Checked {} {} {} Got: {}".format(idx, chld_id, gc_id, do_combine))
                self.checked_gcs.append((idx, gc_id))

                if do_combine:
                    self.lchildren[idx].append([gc_id, new_wt])
                    logi("Added {} to {}: {}".format(gc_id, idx, self.lchildren[idx]))

                igrandchild += 1

            ichild += 1

        logd("Processed {}".format(idx))
        self.processed.append(idx)

    @property
    def parents_info(self):
        info = ''
        for i, parents in enumerate(self.lparents):
            info += "\n{}: ".format(i)
            for j, p in parents:
                info += "{}({},{:.3f}); ".format(j, p.best_char, p.strength())
        return info

    def process_tree(self):
        self.process_node(0)
        self.find_parents()
        if logger.isEnabledFor(logging.INFO):
            # logi(self.parents_info)
            logi(str(self))

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

        logd("SP in {}".format(node))
        if node in self.path_strength_till:
            logd("SP already did {}".format(node))

        elif len(self.lparents[node]) == 0:
            self.path_strength_till[node] = 0, [node]
            logd("SP at root {}".format(node))

        else:
            best_strength, best_path = -np.inf, []
            for parent, wt in self.lparents[node]:
                strength, path_till = self.strongest_path(parent)
                strength += wt.strength()
                if strength > best_strength:
                    best_strength = strength
                    best_path = path_till

            self.path_strength_till[node] = best_strength, best_path + [node]
            logi("SP\tBest path at node {} is {}(+{})".format(
                node, best_path, best_strength))

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