import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.weight'] = 600
plt.rcParams['lines.markersize'] = 1

font = {
    'family': 'Roboto',
    'weight': 600
}

# Argv[1] = maximum exposure
# Argv[2] = no balance
# Argv[3] = balance 10:1

data = [x for x in csv.reader(open(sys.argv[1],'r'))][0]
data2 = []
for x in data:
    try:
        data2.append(float(x))
    except:
        pass
dataX = [x for x in range(len(data2))]

data = [x for x in csv.reader(open(sys.argv[2],'r'))][0]
data3 = []
for x in data:
    try:
        data3.append(int(x))
    except:
        pass

s_X = len(data2)
s_Y = len(data3)

if (s_X > s_Y):
    data2 = data2[:s_Y]
    dataX = [x for x in range(len(data3))]
else:
    data3 = data3[:s_X]
    dataX = [x for x in range(len(data2))]

data = [x for x in csv.reader(open(sys.argv[3],'r'))][0]
data4 = []
for x in data:
    try:
        data4.append(int(x))
    except:
        pass

# fig = plt.figure(figsize =(9, 9))
fig, ax = plt.subplots(figsize =(10, 5), constrained_layout=True)

#plt.xticks(rotation=90)
plt.ylabel('Exposure (#)', fontdict=font)
plt.ylim(0,750000)
plt.xlim(-.5,260)
plt.yticks(np.arange(0,750000,50000))
plt.xticks(np.arange(0,260,10))
plt.title('Malware Exposure over time', fontdict=font)
plt.xlabel('Elapsed Epochs (#)', fontdict=font)

plt.plot(dataX, data2, label="No Detection", marker='o', linestyle='--') #markersize=1)
plt.plot(dataX, data3, label="No Balance-No Retrain", marker='o', linestyle='--') #markersize=1)
plt.plot(dataX, data4, label="10: Balance-No Retrain", marker='s', linestyle='--') #markersize=1)

plt.legend(loc="upper left")
plt.grid(axis='both')

plt.tight_layout()

plt.show()

#fig.savefig('profile.pdf')  
