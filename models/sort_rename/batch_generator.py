import glob, random
import pandas as pd
import numpy as np
from pythreader import Primitive

class BatchGenerator(Primitive):
    
    def __init__(self, in_dir, ntest = 0, nvalidate = 0, maxlen = 70000):
        Primitive.__init__(self)
        self.InDir = in_dir
        files = sorted(glob.glob(in_dir + "/event*.hd5"))
        self.TestFiles = files[:ntest]
        self.ValidateFiles = files[ntest:ntest+nvalidate]
        self.TrainFiles = files[ntest+nvalidate:]
        self.NTrain = len(self.TrainFiles)
        self.NTest = ntest
        self.NValidate = nvalidate
        self.Bits = ["ipid_%d" % (i,) for i in range(16)]
        self.MaxLen = maxlen
        
    def loadEvent(self, f):
        store = pd.HDFStore(f)
        hits = store["train_hits"]
        x = hits[["x", "y", "z"]].values
        y = hits[self.Bits].values
        #print "loaded event(%s) x:%s, y:%s" % (f, x.shape, y.shape)
        store.close()
        #return x[:10000, :], y[:10000, :]
        #print x.shape, y.shape
        return x, y
                
    def loadSet(self, fset):
        xs = []
        ys = []
        for f in fset:
            x, y = self.loadEvent(f)
            xs.append(x)
            ys.append(y)
        n = min([len(x) for x in xs])
        return np.array([x[:n] for x in xs]), np.array([y[:n] for y in ys])
    
    def testSet(self):
        return self.loadSet(self.TestFiles)
        
    def validateSet(self):
        return self.loadSet(self.ValidateFiles)
        
    @synchrinized
    def trainGenerator(self, mbsize=10, files = None):
        if files is None:
            files = self.TrainFiles
        for i in xrange(0, mbsize, len(files)):
            fset = files[i:i+mbsize]
            x, y = self.loadSet(fset)
            yield x, y

    @synchrinized
    def infiniteTrainGenerator(self, mbsize=10, shuffle=True):
        files = self.TrainFiles[:]
        while True:
            for x, y in self.trainGenerator(files=files, mbsize=mbsize):
                yield x, y
            random.shuffle(files)