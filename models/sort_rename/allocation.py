from keras import backend as K
from batch_generator import BatchGenerator

import getopt, sys

dataProvider = BatchGenerator("", nvalidate = 10)

f = sys.argv[1]

x, y = dataProvider.loadEvent(f)

print x.shape, y.shape
print len(x.data), len(y.data)

xt = K.variable(value=x)
yt = K.variable(value=y)

print "allocated"