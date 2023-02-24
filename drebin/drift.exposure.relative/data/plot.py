import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

# Large fonts, better to read in single column papers
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.weight'] = 600
plt.rcParams['font.size'] = 16
# had to decrease line size to differentiate points when close (is there a better solution?)
plt.rcParams['lines.markersize'] = 1

font = {
    'family': 'Roboto',
    'weight': 600,
    'size': 16
}

# Trying to generate reproducible figs from the paper, so read from the same file
files = ["nodetection.csv","best.csv","ddm.csv", "eddm.csv"] 

# We need to specify labels, in the correct order
# Let's try to put labels in the exposure order, for easing the reading
labels = ["50%/6:1", "50%/6:1 (DDM)", "50%/6:1 (EDDM)"] 

def get_arrays(filename,base=None):
    data = [x for x in csv.reader(open(filename,'r'))][0]
    data2 = []
    for idx, x in enumerate(data):
        try:
            # sometimes need to convert data types
            # in most cases, we do not need this conversion, just return the first array
            if base is None:
                data2.append(int(x))
            else:
                data2.append(100.0*int(x)/float(base[idx]))
        except:
            pass
    dataX = [x for x in range(len(data2))]
    return dataX, data2

vectors = []
# open each filename and get the exposure rates for each one
ref = get_arrays(files[0])
for i in files[1:]:
    vectors.append(get_arrays(i,ref[1]))

# Plot configuration, boring stuff
fig, ax = plt.subplots(figsize =(10, 5), constrained_layout=True)
#plt.xticks(rotation=90)
plt.ylabel('Relative Exposure (%)', fontdict=font)
plt.ylim(40,101)
plt.xlim(-.5,260)
plt.yticks(np.arange(40,101,5))
plt.xticks(np.arange(0,270,30))
plt.title('Malware (Relative) Exposure over time', fontdict=font)
plt.xlabel('Elapsed Epochs (#)', fontdict=font)

# plot the points previously read from files
for i, v in enumerate(vectors):
    # we must ensure all Xs are equals
    # Plot X,Y, and associated labels
    plt.plot(v[0], v[1], label=labels[i], marker='o', linestyle='--') #markersize=1)

plt.legend(loc="lower center",ncol=5)
plt.grid(axis='both')

plt.tight_layout()

plt.show()

fig.savefig('drift.exṕosure.relative.pdf')  
