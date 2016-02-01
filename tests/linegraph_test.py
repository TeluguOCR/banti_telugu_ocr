from random import random

from banti.linegraph import LineGraph


class Weight():
    def __init__(self, val):
        self.val = val

    def combine(self, other):
        return random() < .3, Weight(int(100*random())+(self.val+other.val)//2)

    def strength(self):
        return self.val

    def __repr__(self):
        return "{}".format(self.val)

weights = [Weight(val) for val in range(10, 80, 10)]

print(list(enumerate(weights)))
lgraph = LineGraph(weights)

print(lgraph.lchildren)
print(lgraph)

lgraph.process_tree()

print(lgraph)

paths = lgraph.get_paths()
for path in paths:
    print(path, lgraph.path_strength(path))

print("Strongest Path: ", lgraph.strongest_path())