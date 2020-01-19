import sys
import math
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
pp = np.array([np.array(['usual', 'pretentious', 'great_pret']), np.array(['proper', 'less_proper', 'improper', 'critical', 'very_crit']), np.array(['complete', 'completed', 'incomplete', 'foster']), np.array(['1', '2', '3', 'more']), np.array(['convenient', 'less_conv', 'critical']), np.array(['convenient', 'inconv']), np.array(['nonprob', 'slightly_prob', 'problematic']), np.array(['recommended', 'priority', 'not_recom']),np.array(['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])])

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

def model(data, test, para):
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
	if (para[0] == 1):
		n = len(data)
		m = int (para[-2]) 
		k = int (para[3])
		st = int (n/k)
		z = [0.0]*(1000)
		for i in range(0,m):
			for j in range(0,st):
				x = data[j*k:(j+1)*k,:].dot(w)
				y = np.exp(x)/(np.sum(np.exp(x), axis = 1)).reshape(len(x),1)
				w = w + para[1]*(data[j*k:(j+1)*k,:].T).dot(datay[j*k:(j+1)*k,:] - y)/(k)
			z[i] = -1*np.sum(np.multiply(datay,np.log(sm(data.dot(w)))))/n
			sa = data.dot(w)
			e = np.sum(np.linalg.norm(datay - np.exp(sa)/(np.sum(np.exp(sa), axis = 1)).reshape(len(sa),1), axis = 1))/n
			if (e < 0.01):
			    break
		y = sm(test.dot(w))
		y = reverse(y)
		return(z,y)


	if (para[0] == 2):
		n = len(data)
		m = int (para[-2]) 
		k = int (para[3])
		st = int (n/k)
		z = [0.0]*(1000*st)
		for i in range(0,m):
			for j in range(0,st):
				x = data[j*k:(j+1)*k,:].dot(w)
				y = np.exp(x)/(np.sum(np.exp(x), axis = 1)).reshape(len(x),1)
				w = w + para[1]/(np.sqrt(i))*(data[j*k:(j+1)*k,:].T).dot(datay[j*k:(j+1)*k,:] - y)/(k)
				z[(i-1)*st + j] = -1*np.sum(np.multiply(datay,np.log(sm(data.dot(w)))))/n
			sa = data.dot(w)
			e = np.sum(np.linalg.norm(datay - np.exp(sa)/(np.sum(np.exp(sa), axis = 1)).reshape(len(sa),1), axis = 1))/n
			if (e < 0.01):
			    break
		y = sm(test.dot(w))
		y = reverse(y)
		return(z,y)



data = genfromtxt(sys.argv[1], dtype = str ,delimiter = ",")
test = data[2000:3001,:]
ye = test[:,len(test[0]) - 1]
data= data[0:2000,:]
'''f = plt.figure()


z1 = model(data,test,[1.0,0.1,1000,200])
z2 = model(data,test,[1.0,0.01,1000,200])
z3 = model(data,test,[1.0,0.001,1000,200])
plt.plot(np.arange(1,len(z1[0])),z1[0][0:len(z1[0])-1], label = "0.1")
plt.plot(np.arange(1,len(z2[0])),z2[0][0:len(z2[0]) -1], label = "0.01")
plt.plot(np.arange(1,len(z3[0])),z3[0][0:len(z2[0]) - 1], label = "0.001")
t =confusion_matrix(ye, z1[1], labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])
print(t)
f1 = f1_score(ye, z1[1], average='macro')
f2 = f1_score(ye, z1[1], average='micro')
print(f1,f2)
plt.title("loss at different eta for constant leraning rate")
plt.legend()
plt.xlabel("iterations") 
plt.ylabel('loss')
f.savefig("fooe.pdf", bbox_inches='tight')'''

'''f = plt.figure()


z1 = model(data,test,[1.0,0.1,1000,500])
z2 = model(data,test,[1.0,0.1,1000,1000])
z3 = model(data,test,[1.0,0.1,1000,1500])
plt.plot(np.arange(1,len(z1[0])),z1[0][0:len(z1[0])-1], label = "300")
plt.plot(np.arange(1,len(z2[0])),z2[0][0:len(z2[0]) -1], label = "400")
plt.plot(np.arange(1,len(z3[0])),z3[0][0:len(z3[0]) - 1], label = "500")
t =confusion_matrix(ye, z1[1], labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])
print(t)
f1 = f1_score(ye, z1[1], average='macro')
f2 = f1_score(ye, z1[1], average='micro')
print(f1,f2)
plt.title("loss at different b for constant leraning rate")
plt.legend()
plt.xlabel("iterations") 
plt.ylabel('loss')
f.savefig("foob.pdf", bbox_inches='tight')

'''
f = plt.figure()


z1 = model(data,test,[2.0,0.1,1000,200])
z2 = model(data,test,[2.0,0.01,1000,200])
z3 = model(data,test,[2.0,0.001,1000,200])
plt.plot(np.arange(1,len(z1[0])),z1[0][0:len(z1[0])-1], label = "0.1")
plt.plot(np.arange(1,len(z2[0])),z2[0][0:len(z2[0]) -1], label = "0.01")
plt.plot(np.arange(1,len(z3[0])),z3[0][0:len(z2[0]) - 1], label = "0.001")
t =confusion_matrix(ye, z1[1], labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])
print(t)
f1 = f1_score(ye, z1[1], average='macro')
f2 = f1_score(ye, z1[1], average='micro')
print(f1,f2)
plt.title("loss at different eta for adaptive leraning rate")
plt.legend()
plt.xlabel("iterations") 
plt.ylabel('loss')
f.savefig("fooea.pdf", bbox_inches='tight')


f = plt.figure()


z1 = model(data,test,[2.0,0.1,1000,500])
z2 = model(data,test,[2.0,0.1,1000,1000])
z3 = model(data,test,[2.0,0.1,1000,1500])
plt.plot(np.arange(1,len(z1[0])),z1[0][0:len(z1[0])-1], label = "0.1")
plt.plot(np.arange(1,len(z2[0])),z2[0][0:len(z2[0]) -1], label = "0.01")
plt.plot(np.arange(1,len(z3[0])),z3[0][0:len(z2[0]) - 1], label = "0.001")
t =confusion_matrix(ye, z1[1], labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])
print(t)
f1 = f1_score(ye, z1[1], average='macro')
f2 = f1_score(ye, z1[1], average='micro')
print(f1,f2)
plt.title("loss for different b for adaptive leraning rate")
plt.legend()
plt.xlabel("iterations") 
plt.ylabel('loss')
f.savefig("foobe.pdf", bbox_inches='tight')
'''para[1.0,0.1,1000,300]
para[1.0,0.1,1000,400]
para[1.0,0.1,1000,500]
para[2.0,0.1,1000,200]
para[2.0,0.01,1000,200]
para[2.0,0.001,1000,200]
para[2.0,0.1,1000,300]
para[2.0,0.1,1000,400]
para[2.0,0.1,1000,500]'''

