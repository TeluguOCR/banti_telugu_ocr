from collections import defaultdict
from linegraph import LineGraph
import logging

log =  logging.info if False else print

class GramGraph(LineGraph):
    ngram = lambda *_: None

    def __init__(self, line_bantries):
        super().__init__(line_bantries)
        self.paths_till = defaultdict(dict)

    def top_ngram_paths(self, node=None):

        if node is None:
            node = len(self.lchildren)-1

        log("IN {}".format(node))
        if node in self.paths_till:
            log("Already did {}".format(node))

        elif len(self.lparents[node]) == 0:
            self.paths_till[node][" "] = 0, 0, 0
            log("At root {}".format(node))

        else:
            for parent, bantry in self.lparents[node]:
                paths_till_parent = self.top_ngram_paths(parent)

                for key, val in paths_till_parent.items():
                    end = key[-1]
                    likli0, prior0, post0 = val
                    for char, likli in bantry.likelies:
                        prior = self.ngram(key + (char,))
                        post = post0 + prior + likli
                        if (end, char) in self.paths_till[node] and \
                                        self.paths_till[node][end, char][-1] > post:
                                continue
                        self.paths_till[node][(end, char)] = likli0 + likli, prior + prior0, post

        return self.paths_till[node]