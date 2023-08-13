import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys
import IPython

# Large fonts, better to read in single column papers
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.weight'] = 600
plt.rcParams['font.size'] = 16

# had to decrease line size to differentiate points when close (is there a better solution?)
plt.rcParams['lines.markersize'] = 3

font = {
    'family': 'Roboto',
    'weight': 600,
    'size': 16
}

# Trying to generate reproducible figs from the paper, so read from the same file
#files = ["ddm_71.csv", "pseudo71.csv", "ddm3.csv"] 
files = ["ddm_71.csv","pseudo71.csv", "ddm3.csv"] 
oracle_files = ["drift71a.csv", "drift71.csv", "drift0.csv"]
pseudo_files = ["drift71b.csv", "drift71p.csv", "drift0p.csv"]


# We need to specify labels, in the correct order
# Let's try to put labels in the exposure order, for easing the reading
labels = ["DDM+71 epochs (DDM/Partial)" ,"DDM+71 epochs (Pseudo)", "50%/6:1 (DDM/Partial)"] 

def get_arrays(filename):
    data = [x for x in csv.reader(open(filename,'r'))][0]
    data2 = []
    for x in data:
        try:
            # sometimes need to convert data types
            # in most cases, we do not need this conversion, just return the first array
            data2.append(int(x))
        except:
            pass
    dataX = [x for x in range(len(data2))]
    return dataX, data2

vectors = []
# open each filename and get the exposure rates for each one
for idx, i in enumerate(files):
    a = get_arrays(i)
    b = get_arrays(oracle_files[idx])
    w = get_arrays(pseudo_files[idx])
    c = []
    d = []
    e = []
    f = []
    g = []
    h = []
    for j in range(len(a[0])):
        if b[1][j] == 1:
            print("positive")
            c.append(j)
            d.append(a[1][j])
            e.append(250)
    for j in range(len(a[0])):
        if w[1][j] == 1:
            print("positive")
            f.append(j)
            g.append(a[1][j])
            h.append(250)

    vectors.append([a[0], a[1], c, d, e, f, g, h])

# Plot configuration, boring stuff
fig, ax = plt.subplots(figsize =(10, 5), constrained_layout=True)
#plt.xticks(rotation=90)
plt.ylabel('Exposure (#)', fontdict=font)
plt.ylim(0,750001)
plt.xlim(-.5,260)
plt.yticks(np.arange(0,750001,50000))
plt.xticks(np.arange(0,260,30))
plt.title('Malware Exposure over time', fontdict=font)
plt.xlabel('Elapsed Epochs (#)', fontdict=font)

# plot the points previously read from files
for i, v in enumerate(vectors):
    # we must ensure all Xs are equals
    # Plot X,Y, and associated labels
    plt.plot(v[0], v[1], label=labels[i], marker='o', linestyle='--', zorder=1) #markersize=1)
    if i == 1:
        plt.scatter(v[2],v[3],label="Oracle",marker='X', sizes = v[4], zorder=2)
        plt.scatter(v[5],v[6],label="Pseudo",marker='v', sizes = v[4], zorder=3)
    if i == 0:
        plt.scatter(v[2],v[3],label="Oracle",marker='*', sizes = v[4], zorder=2)
    if i == 2:
        plt.scatter(v[2],v[3],label="Oracle",marker='o', sizes = v[4], zorder=2)


plt.legend(loc="upper left",ncol=1)
plt.grid(axis='both')

plt.tight_layout()

plt.show()

fig.savefig('pseudp500-71-drift.pdf')  
