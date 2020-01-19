import numpy as np
import sys
from numpy import genfromtxt

#read the file as
data =genfromtxt(sys.argv[1],delimiter = ",")
datay = data[:,-1:]
data = np.delete(data, len(data[0]) - 1, axis = 1)
print(np.amax(data))
data = data
test =genfromtxt(sys.argv[2],delimiter = ",")
test = np.delete(test, len(test[0]) - 1, axis = 1)

var = np.var(data, axis = 0)
unimp = []
for i in range(0,len(data[0])):
	if (var[i] == 0.00):
		unimp.append(i)
data = np.delete(data, unimp, axis = 1)
data = np.append(np.array([[1]]*len(data)),data, axis = 1)
test = np.delete(test, unimp, axis = 1)
test = np.append( np.array([[1]]*len(test)),test, axis = 1)
print(data.shape, datay.shape, test.shape)

class svm:

  def __init__(self,no_classses, no_of_features):
    self.n = no_classses
    self.m = no_of_features
    self.w = [np.array([[-1.0]]*(no_of_features))]*int((no_classses)*(no_classses - 1)/2)
    #self.w = [np.random.rand(no_of_features,1)*2]*int((no_classses)*(no_classses - 1)/2)
    self.tsvm = int((no_classses)*(no_classses - 1)/2)
    self.b = np.array([[0,0]]*self.tsvm)
    y = 0
    for j in range(0,10):
      for i in range(j+1,10):
        self.b[y][0] = j
        self.b[y][1] = i
        y += 1

  def tm(self,x,y,i):
    w = self.w[i]
    t = ((w.T).dot(x))*y
    t = (t/(2*np.absolute(t)) + 0.5)*y
    return(y)

  def infer(self,x):
    an = np.array([[0]]*len(x))
    for k in range(0, len(x)):
      temp = np.array([0]*self.n)
      m = 0
      ans = 0
      for i in range(0,self.tsvm):
        a = x[k].dot(self.w[i])
        if(a > 0):
          temp[self.b[i][0]] += 1
          if(temp[self.b[i][0]] >= m):
            m = temp[self.b[i][0]]
            ans = self.b[i][0]
        else:
          temp[self.b[i][1]] += 1
          if(temp[self.b[i][1]] >= m):
            m = temp[self.b[i][1]]
            ans = self.b[i][1]
      an[k][0] = ans
    return(an)
  
  #now lerning will be per svm
  def learn(self,datax,datay,b,c,al):
    for i in range(0,self.tsvm):
      tempx = []
      tempy = []
      w = self.w[i]
      a1 = self.b[i][0]
      a2 = self.b[i][1]
      for j in range(0,len(datax)):
        if(datay[j] == a1 or datay[j] == a2):
          tempx = tempx + [datax[j]]
          tempy = tempy + [datay[j]]
      tempx = np.array(tempx)
      tempy = np.array(tempy)
      tempy[tempy == a1] = 1.0
      tempy[tempy == a2] = -1.0
      for l in range(0,150):
        for g in range(0,40):
          x = tempx[g*b:(g+1)*b,:]
          y = tempy[g*b:(g+1)*b,:]
          dw = (x.dot(w))*y
          dw[dw <= 1.0] = -1
          dw[dw > 1.0] = 0
          dw = w + c*(x.T).dot(dw*y)/b
          bo = w[0][0]
          w = w - dw*al
          w[0][0] = w[0][0] + al*bo
      self.w[i] = w

d = svm(10,len(data[0]))
x = np.array([[-1]*10]*2)
d.learn(data,datay,0,2.0,0.008)
z = 0
r = d.infer(test)
np.savetxt(sys.argv[3] , r)