import sys
import math
import numpy as np
from numpy import genfromtxt
data = genfromtxt(sys.argv[1], dtype = str ,delimiter = ",")
test = genfromtxt(sys.argv[2], dtype = str ,delimiter = ",")
pp = np.array([np.array(['usual', 'pretentious', 'great_pret']), np.array(['proper', 'less_proper', 'improper', 'critical', 'very_crit']), np.array(['complete', 'completed', 'incomplete', 'foster']), np.array(['1', '2', '3', 'more']), np.array(['convenient', 'less_conv', 'critical']), np.array(['convenient', 'inconv']), np.array(['nonprob', 'slightly_prob', 'problematic']), np.array(['recommended', 'priority', 'not_recom']),np.array(['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])])
def change(x):
	ans = [0.0]*32
	ans[np.where(pp[0] == x[0])[0][0]] = 1
	ans[np.where(pp[1] == x[1])[0][0] + 3] = 1
	ans[np.where(pp[2] == x[2])[0][0] + 8] = 1
	ans[np.where(pp[3] == x[3])[0][0] + 12] = 1
	ans[np.where(pp[4] == x[4])[0][0] + 16] = 1
	ans[np.where(pp[5] == x[5])[0][0] + 19] = 1
	ans[np.where(pp[6] == x[6])[0][0] + 21] = 1
	ans[np.where(pp[7] == x[7])[0][0] + 24] = 1
	ans[np.where(pp[8] == x[8])[0][0] + 27] = 1
	return(ans)
def changet(x):
	ans = [0.0]*27
	ans[np.where(pp[0] == x[0])[0][0]] = 1
	ans[np.where(pp[1] == x[1])[0][0] + 3] = 1
	ans[np.where(pp[2] == x[2])[0][0] + 8] = 1
	ans[np.where(pp[3] == x[3])[0][0] + 12] = 1
	ans[np.where(pp[4] == x[4])[0][0] + 16] = 1
	ans[np.where(pp[5] == x[5])[0][0] + 19] = 1
	ans[np.where(pp[6] == x[6])[0][0] + 21] = 1
	ans[np.where(pp[7] == x[7])[0][0] + 24] = 1
	return(ans)
def reverse(x):
	t = pp[-1]
	ans = ["d"]*len(x)
	for i in range(0,len(x)):
		ans[i] = t[np.argmax(x[i])]
	return(ans)
def sm(x):
	ans = np.array([[0.0]*5]*len(x))
	for i in range(0,len(x)):
		ans[i] = np.exp(x[i])/np.sum(np.exp(x[i]))
	return(ans)
data1 = np.array([[0.0]*32]*len(data))
test1 = np.array([[0.0]*27]*len(test))
for i in range(0,len(data)):
	data1[i] = change(data[i])
for i in range(0,len(test)):
	test1[i] = changet(test[i])
test = test1
data = data1
datay = data[:,(len(data[0]) - len(pp[-1])):]
data = np.delete(data, np.arange(len(data[0]) - len(pp[-1]),len(data[0]) ), axis = 1)
data = np.append([[1]]*len(data), data, axis = 1)
test = np.append([[1]]*len(test), test, axis = 1)
w = np.array([[0]*5]*len(data[0]))
f = open(sys.argv[3])
z = float (f.readline())
if (z ==3):
	para = np.array([0.0]*5)
	para[0] = z
	para[1:4] = np.array(f.readline().split(',') , dtype = float)
	para[4] = float (f.readline())
else:
	para = np.array([0.0]*3)
	para[0] = z
	para[1] = float (f.readline())
	para[2] = float (f.readline())
if (para[0] == 1):
	n = len(data)
	m = int (para[-1]) + 1
	for i in range(1,m):
	    x = data.dot(w)
	    y = np.exp(x)/(np.sum(np.exp(x), axis = 1)).reshape(len(x),1)
	    e = np.sum(np.linalg.norm(datay - y, axis = 1))/n
	    if(e < 0.01):
		    break
	    w = w + para[1]*(data.T).dot(datay  - y)/(n)
	y = sm(test.dot(w))
	y = reverse(y)
	np.savetxt(sys.argv[5] , w, delimiter = ',')
	f2= open(sys.argv[4],"w+")
	for j in range(0,len(test)):
		f2.write(y[j] + '\n')
	print(w)
	

if (para[0] == 2):
	n = len(data)
	m = int (para[-1]) + 1
	for i in range(1,m):
	    x = data.dot(w)
	    y = np.exp(x)/(np.sum(np.exp(x), axis = 1)).reshape(len(x),1)
	    e = np.sum(np.linalg.norm(datay - y, axis = 1))/n
	    if(e < 0.01):
		    break
	    w = w + para[1]/np.sqrt(i)*(data.T).dot(datay - y)/(n)
	y = sm(test.dot(w))
	y = reverse(y)
	np.savetxt(sys.argv[5] , w, delimiter = ',')
	f2= open(sys.argv[4],"w+")
	for j in range(0,len(test)):
		f2.write(y[j] + '\n')
	print(i)

if (para[0] == 3):
    n = len(data)
    o = len(w)*len(w[0])
    m = int (para[-1]) + 1
    x = data.dot(w)
    y = np.exp(x)/(np.sum(np.exp(x), axis = 1)).reshape(len(x),1)
    zold = -1*np.sum(np.multiply(datay,np.log(sm(y))))/n
    for i in range(1,m):
	    d = (data.T).dot(y - datay)/n
	    a = w - para[1]*d
	    z = data.dot(a)
	    z = np.exp(z)/(np.sum(np.exp(z), axis = 1)).reshape(len(z),1)
	    z = -1*np.sum(np.multiply(datay,np.log(sm(z))))/n
	    q = zold + para[1]*para[2]*(((d.reshape(o,1)).T).dot(d.reshape(o,1)))[0][0]
	    if(z <= q):
	        break
	    para[1] = para[1]*para[3]
    w = np.array([[0]*5]*len(data[0]))
    for i in range(1,m):
	    u = data.dot(w)
	    v = np.exp(u)/(np.sum(np.exp(u), axis = 1)).reshape(len(u),1)
	    e = np.sum(np.linalg.norm(datay - v, axis = 1))/n
	    if(e < 0.01):
	        break
	    w = w + para[1]*(data.T).dot(datay - v)/(n)
    y = sm(test.dot(w))
    y = reverse(y)
    np.savetxt(sys.argv[5] , w, delimiter = ',')
    f2= open(sys.argv[4],"w+")
    for j in range(0,len(test)):
        f2.write(y[j] + '\n')
    print(i)