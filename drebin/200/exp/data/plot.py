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
files = ["nodetection-200.csv", "simple-detection.csv", "delay-160.csv", "delay-100.csv",  "drift-partial.csv", "drift.csv" ] #, "detection-thresh.csv", "delay-60.csv", "delay-40.csv", "delay10.csv", "drift-partial.csv", "1000-drift.csv"] #, "drift-ideal.csv"]

# We need to specify labels, in the correct order
# Let's try to put labels in the exposure order, for easing the reading
labels = ["No Detection", "50%/6:1", "50%/6:1 (DDM+160)", "50%/6:1 (DDM+100)", "50%/6:1 (DDM/Partial)", "50%/6:1 (DDM)"] #, "50%/6:1 (DDM+60)", "50%/6:1 (DDM+40)", "50%/6:1 (DDM+10)", "50%/6:1 (DDM/Partial)", "50%/6:1 (DDM)"] 

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
#IPython.embed()
max_X = max(vectors[0][0])
X_tics = round(max_X/10)
max_Y = max(vectors[0][1])
Y_tics = round(max_Y/10)
if max_Y % Y_tics < 0.5 * Y_tics:
    max_Y = max_Y + (max_Y % Y_tics) + Y_tics
else:
    max_Y = max_Y + (max_Y % Y_tics)
if max_X % X_tics < 0.5 * X_tics:
    max_X = max_X + (max_X % X_tics) + X_tics
else:
    max_X = max_X + (max_X % X_tics)

plt.ylim(0,max_Y)
plt.xlim(-.5,max_X)
plt.yticks(np.arange(0,max_Y,Y_tics))
plt.xticks(np.arange(0,max_X,X_tics))
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

fig.savefig("plot_200.pdf")  
