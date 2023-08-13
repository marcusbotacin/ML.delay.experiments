# fix files computed with 1/6 undue overhead
import sys
f = open(sys.argv[1],'r').read().strip().split(',')
for item in f:
    try:
        val = 5 * int(item) / 6
        print("%d," % val),
    except:
        pass
