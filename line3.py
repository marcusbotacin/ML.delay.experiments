import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.weight'] = 600

font = {
    'family': 'Roboto',
    'weight': 600
}

data = [x for x in csv.reader(open(sys.argv[1],'r'))][0]
data2 = []
for x in data:
    try:
        data2.append(float(x))
    except:
        pass
dataX = [x for x in range(len(data2))]

# fig = plt.figure(figsize =(9, 9))
fig, ax1 = plt.subplots(figsize =(10, 5), constrained_layout=True)

#plt.xticks(rotation=90)
#ax1.ylabel('Branch Misses (#)', fontdict=font)
#ax1.ylim(0.110,10)
#ax1.xlim(-.5,300)
#ax1.yticks(np.arange(0,110,10))
#ax1.xticks(np.arange(0,300,10))
#ax1.title('Behavioral Profiles', fontdict=font)
#ax1.xlabel('Elapsed Time (s)', fontdict=font)

ax1.plot(dataX, data2, label="Precision", marker='o', linestyle='--') #markersize=1)
#plt.plot(freq1, freq3, label="Malware", marker='s', linestyle='--') #markersize=1)

ax2 = ax1.twinx()

data = [x for x in csv.reader(open(sys.argv[2],'r'))][0]
data2 = []
for x in data:
    try:
        data2.append(float(x))
    except:
        pass
dataX = [x for x in range(len(data2))]

ax2.plot(dataX, data2, label="Precision", marker='o', linestyle='--') #markersize=1)

plt.legend(loc="upper right")
plt.grid(axis='both')

plt.tight_layout()

plt.show()

fig.savefig('profile.pdf')  
