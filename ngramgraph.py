from collections import defaultdict
import logging

from linegraph import LineGraph


logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug

class PathNode():
    order = 0
    def __init__(self, likli=0, prior=0, chars=(" ",)):
        self.likli, self.prior = likli, prior
        self.chars = chars

    @property
    def key(self):
        return self.chars[-self.order:]

    @property
    def post(self):
        return self.likli + self.prior

    def __add__(self, other):
        return PathNode(self.likli + other.likli,
                        self.prior + other.prior,
                        self.chars + other.chars)

    def __str__(self):
        return "{} L{:.3f}+R{:.3f}=T{:.3f}".format('|'.join(self.chars),
                                    self.likli, self.prior, self.post)

class GramGraph(LineGraph):
    ngram = lambda *_: 0

    def __init__(self, line_bantries):
        super().__init__(line_bantries)
        self.paths_till = defaultdict(dict)  # paths_till[node]['a', 'b']
                                             # second key is a tuple/list of length ngram.n-1

    @classmethod
    def set_ngram(cls, ng):
        cls.ngram = ng
        PathNode.order = ng.n - 1

    def find_top_ngram_paths(self, node=None):
        if node is None:
            node = self.last_node

        logd("Graming at {}".format(node))
        if node in self.paths_till:
            logd("Already grammed {}".format(node))

        elif len(self.lparents[node]) == 0:
            ppn = PathNode()
            self.paths_till[node][ppn.key] = ppn
            logd("Gramming root {}".format(node))

        else:
            for parent, bantry in self.lparents[node]:
                paths_till_parent = self.find_top_ngram_paths(parent)
                logi(bantry.strlikelies)

                for ppn_key, ppn in paths_till_parent.items():
                    for char, likli in bantry.likelies:
                        prior = self.ngram(ppn_key + (char,))
                        pn = ppn + PathNode(likli, prior, (char,))
                        if pn.key in self.paths_till[node]:
                            if self.paths_till[node][pn.key].post > pn.post:
                                continue
                        self.paths_till[node][pn.key] = pn

            if logger.isEnabledFor(logging.INFO):
                logi("Final Paths for node {}\n{}".format(
                    node, self.top_pathnodes_at(node)))

        return self.paths_till[node]

    @property
    def top_final_pathnode(self):
        try:
            return self.top_final_pathnode_
        except AttributeError:
            self.top_final_pathnode_ = max(
                self.find_top_ngram_paths().values(),
                key=lambda p: p.post)
            return self.top_final_pathnode

    def top_pathnodes_at(self, node, n=5, as_str=True):
        top = sorted(self.paths_till[node].values(),
                     key=lambda p:p.post,
                     reverse=True)[:n]
        if as_str:
            top = "\n".join([str(v) for v in top])
        return top

    def get_best_str(self, join=""):
        return join.join(self.top_final_pathnode.chars)

    def get_path_chars(self, path, join=None):
        ret = []
        for i in range(len(path)-1):
            for child, bantry in self.lchildren[path[i]]:
                if child == path[i+1]:
                    ret.append(bantry.best_char)
                    break
            else:
                raise ValueError("Path not found in graph: {}".format(path))

        if join:
            ret = join.join(ret)
        return ret

    def get_best_apriori_str(self, join=None):
        liklihood, most_likely = self.strongest_path()
        return self.get_path_chars(most_likely, join)

if __name__ == "__main__":
    import sys
    from scaler import ScalerFactory
    from bantry import Bantry, BantryFile
    from classifier import Classifier
    from ngram import Ngram

    nnet_file = sys.argv[1] if len(sys.argv) > 1 else "library/nn.pkl"
    banti_file_name = sys.argv[2] if len(sys.argv) > 2 else "sample_images/praasa.box"
    scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
    labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"
    ngram_file = "library/mega.123.pkl"

    Bantry.scaler = ScalerFactory(scaler_prms_file)
    Bantry.classifier = Classifier(nnet_file, labellings_file, logbase=1)
    bf = BantryFile(banti_file_name)

    ngram = Ngram(ngram_file)
    GramGraph.set_ngram(ngram)

    for linenum in range(bf.num_lines):
        print('*' * 80)
        bantires = bf.get_line_bantires(linenum)
        gramgraph = GramGraph(bantires)
        gramgraph.process_tree()
        gramgraph.find_top_ngram_paths()
        for node, children in enumerate(gramgraph.lchildren):
            print(gramgraph.top_pathnodes_at(node, 1))
        print(gramgraph.get_best_str('|'))
        print(gramgraph.get_best_apriori_str('|'))
