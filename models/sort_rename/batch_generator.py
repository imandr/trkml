import glob, random
import pandas as pd
import numpy as np


class BatchGenerator(object):
    
    def __init__(self, in_dir, ntest = 0, nvalidate = 0, maxlen = 70000):
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
            xs.append(x[:self.MaxLen])
            ys.append(y[:self.MaxLen])
        return np.array(xs), np.array(ys)
    
    def testSet(self):
        return self.loadSet(self.TestFiles)
        
    def validateSet(self):
        return self.loadSet(self.ValidateFiles)
        
    def trainGenerator(self, files = None):
        if files is None:
            files = self.TrainFiles
        for f in files:
            x, y = self.loadEvent(f)
            yield x.reshape((1,)+x.shape), y.reshape((1,)+y.shape)
            
    def infiniteTrainGenerator(self, shuffle=True):
        files = self.TrainFiles[:]
        while True:
            for x, y in self.trainGenerator(files):
                yield x, y
            random.shuffle(files)