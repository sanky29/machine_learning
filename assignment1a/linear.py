import numpy as np
from numpy import genfromtxt
import sys
import sklearn
from sklearn import linear_model
import operator

if (sys.argv[1] == 'a'):
	trainx = np.array(genfromtxt(sys.argv[2] , delimiter = ','))
	test_data = np.array(genfromtxt(sys.argv[3] , delimiter = ','))
	trainy = np.array([[0.0]]*len(trainx))
	for i in range(0,len(trainx)):
		trainy[i][0] = trainx[i][-1]
		trainx[i][-1] = 1
	w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(trainx),trainx)),np.transpose(trainx)), trainy)
	f1= open(sys.argv[5],"w+")
	f1.write(str(w[-1][0]) + "\n")
	for i in range(0,len(w) -1):
	     f1.write(str(w[i][0]) + "\n")
	temp = [[1]]*len(test_data)
	test_data = np.append(test_data,temp, axis = 1)
	f2= open(sys.argv[4],"w+")
	for i in range(0,len(test_data)):
		f2.write(str((np.dot(np.array([test_data[i]]), w))[0][0]) + "\n")

if (sys.argv[1] == 'b'):
	#read whole data
	train_x = np.array(genfromtxt(sys.argv[2] , delimiter = ','))
	train_y = train_x[:,(len(train_x[0]) - 1):(len(train_x[0]))]
	train_x = np.delete(train_x , len(train_x[0]) - 1, axis = 1)
	train_x = np.append([[1]]*len(train_x) , train_x, axis = 1)
	test_data = np.array(genfromtxt(sys.argv[3] , delimiter = ','))
	test_data = np.append([[1]]*len(test_data),test_data, axis = 1)
	Lambda = np.loadtxt(sys.argv[4])
	Lambda_error = [0.0]*len(Lambda)
	step = int (len(train_x)/10)
	for k in range(0,10):
		start = k*step
		end = (k+1)*step
		#have slice between k*(len(test_data)/10) + (k + 1)*(len(test_data)/10)
		testk = np.append(train_x[0:start],train_x[end: step*10], axis = 0)
		valk = train_x[start:end]
		testky = np.append(train_y[0:start],train_y[end: step*10], axis = 0)
		valky = train_y[start:end]
		for i in range(0, len(Lambda)):
			w = ((np.linalg.inv((testk.T).dot(testk) + Lambda[i]*np.identity(len(testk[0])))).dot(testk.T)).dot(testky)
			valky =(valky - np.dot(valk, w))/len(valky)
			Lambda_error[i] = Lambda_error[i] + np.dot(np.transpose(valky), valky)[0][0]
	l = np.where(Lambda_error == np.amin(Lambda_error))[0][0]
	w = ((np.linalg.inv((train_x.T).dot(train_x) + Lambda[l]*np.identity(len(train_x[0])))).dot(train_x.T)).dot(train_y)
	np.savetxt(sys.argv[6] , w)
	np.savetxt(sys.argv[5] , np.dot(test_data, w))
	print(Lambda[l])	

if (sys.argv[1] == 'c'):
	#read whole data
	train_x = np.array(genfromtxt(sys.argv[2] , delimiter = ','))
	train_y = train_x[:,(len(train_x[0]) - 1):(len(train_x[0]))]
	train_x = np.delete(train_x, len(train_x[0]) -1, axis = 1)
	test_data = np.array(genfromtxt(sys.argv[3] , delimiter = ','))
	for q in range(0,int (len(train_x[0])/2)):
	    train_x = np.append(train_x, (train_x[:,2*q:(2*q+1)]+ train_x[:,(2*q+1):2*(q+1)]), axis=1)
	    test_data = np.append(test_data, (test_data[:,2*q:(2*q+1)]+ test_data[:,(2*q+1):2*(q+1)]), axis=1)
	Lambda = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
	var = [(0,0)]*len(train_x[0])
	for i in range (0,len(train_x[0])):
		var[i] = (i, np.var(train_x[:,i]))
	var.sort(key = operator.itemgetter(1))
	dele = np.array([0]*215)
	for k in range(0, 215):
		dele[k] = var[k][0]
	train_x = np.delete(train_x , dele, axis = 1)
	test_data = np.delete(test_data , dele, axis = 1)
	test_data = np.append(test_data, np.power(test_data,2) , axis = 1)
	train_x = np.append(train_x, np.power(train_x,2) , axis = 1)
	Lambda_error = [0.0]*len(Lambda)
	step = int (len(train_x)/10)
	last = step*10
	for k in range(0,10):
	    start = k*step
	    end = (k+1)*step
	    testk = np.append(train_x[0:start],train_x[end: last], axis = 0)
	    valk = train_x[start:end]
	    testky = np.append(train_y[0:start],train_y[end: last], axis = 0)
	    valky = train_y[start:end]
	    for i in range(0, len(Lambda)):
		    model = linear_model.LassoLars(alpha=Lambda[i], verbose=False)
		    model.fit(testk, testky)
		    Lambda_error[i] = Lambda_error[i] + sklearn.metrics.mean_squared_error(model.predict(valk), valky)/len(valky)
	l = np.where(Lambda_error == np.amin(Lambda_error))[0][0]
	print(Lambda[l])
	model = linear_model.LassoLars(Lambda[l])
	model.fit(train_x, train_y)
	np.savetxt(sys.argv[4] , model.predict(test_data))
	#w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(trainx),trainx)),np.transpose(trainx)), trainy)
	#f1= open(sys.argv[5],"w+")
	#for i in range(0,len(w)):
	#     f1.write(str(w[i][0]) + "\n")
	#b = np.array([w[-1]])
	#w = np.delete(w, -1,axis = 0)
	#f2= open(sys.argv[4],"w+")
	#for i in range(0,len(test_data)):
	#	f2.write(str(((np.dot(np.array([test_data[i]]), w) + b)[0])[0]) + "\n")
if (sys.argv[1] == 'd'):
	#read whole data
	set =  np.array(genfromtxt(sys.argv[2] , delimiter = ','))
	train_x = set[:27000]
	train_y = train_x[:,(len(train_x[0]) - 1):(len(train_x[0]))]
	train_x = np.delete(train_x , len(train_x[0]) - 1, axis = 1)
	train_x = np.append([[1]]*len(train_x) , train_x, axis = 1)
	test_data = set[27001:30000]
	test_datay = test_data[:,(len(train_x[0]) - 1):(len(train_x[0]))]
	test_data = np.delete(test_data , len(train_x[0]) - 1, axis = 1)
	test_data = np.append([[1]]*len(test_data),test_data, axis = 1)
	Lambda = np.loadtxt(sys.argv[4])
	Lambda_error = [0.0]*len(Lambda)
	step = int (len(train_x)/10)
	for k in range(0,10):
		start = k*step
		end = (k+1)*step
		#have slice between k*(len(test_data)/10) + (k + 1)*(len(test_data)/10)
		testk = np.append(train_x[0:start],train_x[end: step*10], axis = 0)
		valk = train_x[start:end]
		testky = np.append(train_y[0:start],train_y[end: step*10], axis = 0)
		valky = train_y[start:end]
		for i in range(0, len(Lambda)):
			w = ((np.linalg.inv((testk.T).dot(testk) + Lambda[i]*np.identity(len(testk[0])))).dot(testk.T)).dot(testky)
			valky =(valky - np.dot(valk, w))/len(valky)
			Lambda_error[i] = Lambda_error[i] + np.dot(np.transpose(valky), valky)[0][0]
	l = np.where(Lambda_error == np.amin(Lambda_error))[0][0]
	w = ((np.linalg.inv((train_x.T).dot(train_x) + Lambda[l]*np.identity(len(train_x[0])))).dot(train_x.T)).dot(train_y)
	print("L2 for ridge " , ((test_datay - test_data.dot(w)).T).dot(test_datay - test_data.dot(w))/1000)
	w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_x),train_x)),np.transpose(train_x)), train_y)
	print("L2 for simple regression " , ((test_datay - test_data.dot(w)).T).dot(test_datay - test_data.dot(w))/1000)


	
			
	#w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(trainx),trainx)),np.transpose(trainx)), trainy)
	#f1= open(sys.argv[5],"w+")
	#for i in range(0,len(w)):
	#     f1.write(str(w[i][0]) + "\n")
	#b = np.array([w[-1]])
	#w = np.delete(w, -1,axis = 0)
	#f2= open(sys.argv[4],"w+")
	#for i in range(0,len(test_data)):
	#	f2.write(str(((np.dot(np.array([test_data[i]]), w) + b)[0])[0]) + "\n")

