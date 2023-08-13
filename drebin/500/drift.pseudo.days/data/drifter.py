import sys

for line in open(sys.argv[1],'r').read().strip().split("\n"):
    oracle = int(line.split("[")[1].split("]")[0]) == 1
    pseudo = int(line.split("[")[2].split("]")[0]) == 1
    if pseudo:
        print("1,"),
    else:
        print("0,"),
#    if oracle:
#        print("oracle")
#    if pseudo:
#        print("pseudo")
