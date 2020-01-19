import numpy as np
import sys
import numpy as np

f = open(sys.argv[1])
z = float (f.readline())
if (z ==3):
	para = np.array([0.0]*5)
	para[0] = z
	para[1:4] = np.array(f.readline().split(',') , dtype = float)
	para[4] = float (f.readline())
else:
	para = np.array([0]*3)
	para[0] = z
	para[1] = float (f.readline())
	para[2] = float (f.readline())
print(para)