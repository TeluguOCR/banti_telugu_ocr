from theano import function, config, shared
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))

print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()

print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Graph Operations: ", [x.op for x in f.maker.fgraph.toposort()])
print("Check if the graph has GpuArray Operations.")