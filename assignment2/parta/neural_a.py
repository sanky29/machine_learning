'''
just import numpy and other libraries
'''
import numpy as np
import sys
from numpy import genfromtxt
'''
define number of layers and the weights as following
read the data
'''

data =genfromtxt(sys.argv[1],delimiter = ",")
datay = data[:,(len(data[0]) - 1):]
data = np.delete(data, len(data[0]) - 1, axis = 1)
'''
the data has been read now just read parameter file
it : iterations
z : type of update
eta : the learning rate
'''
f = open(sys.argv[2])
z = int (f.readline())
eta = float (f.readline())
it = int (f.readline())
temp = f.readline()
nunits = [int(i) for i in temp.split()]
#add the input and output to nunits
nunits = [len(data[0])] + nunits + [len(datay[0])]
k = int (f.readline())
'''
create weights and units as folowing
'''
layers = [np.zeros((k,nunits[0]))]
weights = []
bias = []

'''
just run for loop and get layers ready
'''
for i in range(1,len(nunits)):
	layers = layers + [np.zeros((k,nunits[i]))]
	weights = weights + [np.zeros((nunits[i-1],(nunits[i])))]
	bias = bias + [np.zeros((1,nunits[i]))]
	
'''
so now we have intialized our model just do one pridiction 
so need to just multiply the matrices
'''
nob = int (len(data)/k)
if (z == 1):
	for u in range(0,it):
		j = u%nob
		layers[0] = data[j*k:(j+1)*k,:]
		for i in range(1,len(layers)):
			layers[i] = layers[i-1].dot(weights[i-1]) + bias[i-1]
			layers[i] = np.exp(layers[i])
			layers[i] = np.divide(layers[i], (layers[i]+1))
		er = datay[j*k:k*(j+1),:] - layers[-1]
		er = np.divide(er, np.multiply(layers[-1], 1 - layers[-1]))
		#this error is dl/dz
		'''jus do once back track
		we have intial error as
		bias_of_ith : bias_of_ith + error_of_ith
		weight_of_ith = weight_of_ith + error_of_ith*last_yi '''
		for i in range(len(weights) -1, -1, -1):
			er = np.multiply(er,(1 - layers[i+1]))
			er = np.multiply(er,layers[i+1])
			bias[i] = bias[i] + eta*np.sum(er, axis = 0)/k
			w = weights[i] + eta*(layers[i].T).dot(er)/k
			er = er.dot(weights[i].T)
			weights[i] = w
else:
	for u in range(0,it):

		j = u%nob
		layers[0] = data[j*k:(j+1)*k,:]
		for i in range(1,len(layers)):
			layers[i] = layers[i-1].dot(weights[i-1]) + bias[i-1]
			layers[i] = np.exp(layers[i])
			layers[i] = np.divide(layers[i], (layers[i]+1))
		er = datay[j*k:k*(j+1),:] - layers[-1]
		er = np.divide(er, np.multiply(layers[-1], 1 - layers[-1]))
		#this error is dl/dz
		'''jus do once back track
		we have intial error as
		bias_of_ith : bias_of_ith + error_of_ith
		weight_of_ith = weight_of_ith + error_of_ith*last_yi '''
		for i in range(len(weights) -1, -1, -1):
			er = np.multiply(er,(1 - layers[i+1]))
			er = np.multiply(er,layers[i+1])
			bias[i] = bias[i] + eta/(np.sqrt(u))*np.sum(er, axis = 0)/k
			w = weights[i] + eta/(np.sqrt(u))*(layers[i].T).dot(er)/k
			er = er.dot(weights[i].T)
			weights[i] = w


'''
the block which writes to file
'''
w = np.array([[0]])
for i in range(0,len(weights)):
	w = np.append(w, bias[i].reshape(len(bias[i][0]),1), axis = 0)
	w =np.append(w,weights[i].reshape(len(weights[i])*len(weights[i][0]), 1),axis =  0)
w = np.delete(w, 0 , axis = 0)
np.savetxt(sys.argv[3] , w)

