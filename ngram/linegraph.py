import random
import logging
log = logging.info

def combine(wt1, wt2):
    return random.random() < .1, (wt1+wt2)//2

class Linetree():
    def __init__(self, wts):
        self.nodes = []
        for i, wt in enumerate(wts):
            self.nodes.append([[i+1, wt]])

        self.nodes.append([])
        self.processed = []
        self.checked = []

    def processnode(self, idx):
        if idx in self.processed:
            log("Already did", idx)
            return

        log("In ", idx)
        ichild = 0
        while ichild < len(self.nodes[idx]):
            chidx, chwt = self.nodes[idx][ichild]
            self.processnode(chidx)
            log("Back in", idx)

            igrandchild = 0
            while igrandchild < len(self.nodes[chidx]):
                grchidx, grchwt = self.nodes[chidx][igrandchild]
                if (idx, grchidx) in self.checked:
                    log("Already checked {} ({}) {}".format(idx, chidx, grchidx))
                    igrandchild += 1
                    continue

                tocombine, newwt = combine(chwt, grchwt)
                self.checked.append((idx, grchidx))
                log("Checking {} {} {} {}".format(idx, chidx, grchidx, tocombine))

                if tocombine:
                    self.nodes[idx].append([grchidx, newwt])
                    log("Addedto {}:{}".format(idx, self.nodes[idx]))

                igrandchild += 1

            ichild += 1

        log("Done with: ", idx)
        self.processed.append(idx)

    def build(self):
        self.processnode(0)

    def get_paths(self, n=0):
        if len(self.nodes[n]) == 0:
            yield [n]

        for ch, wt in self.nodes[n]:
            for sub_path in self.get_paths(ch):
                yield [n] + sub_path

    def pathwt(self, path):
        ret = 0
        for i in range(len(path)-1):
            for child, wt in self.nodes[path[i]]:
                if child == path[i+1]:
                    ret += wt
                    break
            else:
                raise ValueError(str(path))

        return ret

    def __str__(self):
        ret = ""
        for i, children in enumerate(self.nodes):
            ret += "\nNode {}: ".format(i)
            for child in children:
                ret += "{}, ".format(child)
        return ret


if __name__ == "__main__":
    wts = range(10, 80, 10)
    print(list(enumerate(wts)))
    lt = Linetree(wts)
    print(lt.nodes)
    print(lt)
    lt.build()
    print(lt)

    paths = lt.get_paths()
    for path in paths:
        print(path, lt.pathwt(path))