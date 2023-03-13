import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

# Large fonts, better to read in single column papers
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.weight'] = 600
# had to decrease line size to differentiate points when close (is there a better solution?)
plt.rcParams['lines.markersize'] = 1

font = {
    'family': 'Roboto',
    'weight': 600
}

# Trying to generate reproducible figs from the paper, so read from the same file
files = ["threshold.data/drift_threshold90.exp2.csv", "threshold.data/drift_threshold80.exp2.csv", "threshold.data/drift_threshold70.exp2.csv", "threshold.data/drift_threshold60.exp2.csv", "threshold.data/drift_threshold50.exp2.csv"]

# We need to specify labels, in the correct order
# Let's try to put labels in the exposure order, for easing the reading
labels = ["Threshold 90", "Threshold 80", "Threshold 70", "Threshold 60", "Threshold 50"]

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
for i in files:
    vectors.append(get_arrays(i))

# Plot configuration, boring stuff
fig, ax = plt.subplots(figsize =(10, 5), constrained_layout=True)
#plt.xticks(rotation=90)
plt.ylabel('Exposure (#)', fontdict=font)
plt.ylim(0,750000)
plt.xlim(-.5,260)
plt.yticks(np.arange(0,750000,50000))
plt.xticks(np.arange(0,260,10))
plt.title('Malware Exposure over time', fontdict=font)
plt.xlabel('Elapsed Epochs (#)', fontdict=font)

# plot the points previously read from files
for i, v in enumerate(vectors):
    # we must ensure all Xs are equals
    # Plot X,Y, and associated labels
    plt.plot(v[0], v[1], label=labels[i], marker='o', linestyle='--') #markersize=1)

plt.legend(loc="upper left")
plt.grid(axis='both')

plt.tight_layout()

plt.show()

#fig.savefig('profile.pdf')  
