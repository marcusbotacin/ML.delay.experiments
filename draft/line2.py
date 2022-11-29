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
fig, ax = plt.subplots(figsize =(10, 5), constrained_layout=True)

#plt.xticks(rotation=90)
plt.ylabel('Branch Misses (#)', fontdict=font)
plt.ylim(0.110,10)
plt.xlim(-.5,300)
plt.yticks(np.arange(0,110,10))
plt.xticks(np.arange(0,300,10))
plt.title('Behavioral Profiles', fontdict=font)
plt.xlabel('Elapsed Time (s)', fontdict=font)

plt.plot(dataX, data2, label="Precision", marker='o', linestyle='--') #markersize=1)
#plt.plot(freq1, freq3, label="Malware", marker='s', linestyle='--') #markersize=1)

plt.legend(loc="upper right")
plt.grid(axis='both')

plt.tight_layout()

plt.show()

fig.savefig('profile.pdf')  
