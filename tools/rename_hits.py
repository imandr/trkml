import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import seaborn as sns
import math
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import sys

def bits(n, w):
    out = np.zeros((w,), dtype=np.uint8)
    i = 0
    while n:
        out[i] = float(n%2)
        n /= 2
        i += 1
    return out

def bits_array(a, w):
    out = np.zeros((len(a), w), dtype=np.uint8)
    for i, x in enumerate(a):
        j = 0
        while x:
            try:    out[i, j] = x%2
            except:
                print "i=", i, "  x=", x, "   original=", a[i]
                raise
            x /= 2
            j += 1
    return out

def rename_hits(hits, truth):
    hits_truth = hits.set_index("hit_id").join(
            truth.set_index("hit_id"), 
            lsuffix="hit_", 
            rsuffix="truth_")

    hits_truth["r2"] = np.square(hits_truth["x"]) + np.square(hits_truth["y"])
    hits_truth["zabs"] = np.abs(hits_truth["z"])
    
    hits_sorted = hits_truth.sort_values(["z", "r2", "x", "y"]).copy()
    #print hits_sorted.head()
    particles = {}
    inx = 1
    ipids = []
    for i, row in hits_sorted.iterrows():
        pid = long(row["particle_id"])
        if pid == 0:
            ipid = 0
        else:
            ipid = particles.get(pid)
            if ipid is None:
                ipid = inx
                inx += 1
                particles[pid] = ipid
        ipids.append(ipid)
        
    print "max index=", inx

    out = hits_sorted.drop("particle_id", axis=1)
    ipids = np.array(ipids, dtype=np.uint16)
    out["ipid"] = ipids
    bits = bits_array(ipids, 16)
    for bit in range(16):
        out["ipid_%d" % (bit,)] = bits[:, bit]
        
    # normalize x,y,z
    out["x"] = hits_sorted["x"].values/1000.0
    out["y"] = hits_sorted["y"].values/1000.0
    out["z"] = hits_sorted["z"].values/3000.0

    #print out.columns
    
    return out


dir_path = sys.argv[1]
out_dir = sys.argv[2]

for event_id, hits, truth in load_dataset(dir_path, parts = ["hits", "truth"]):
    hd5_file = "%s/event%d.hd5" % (out_dir, event_id)
    try:	os.remove(hd5_file)
    except:	pass

    renamed = rename_hits(hits, truth)

    hits = len(renamed)
    particles = np.max(renamed["ipid"])

    hd5_store = pd.HDFStore(hd5_file)
    hd5_store["train_hits"] = renamed
    hd5_store.close()
    print "event %s done. %d hits, %d particles" % (event_id,hits,particles)

