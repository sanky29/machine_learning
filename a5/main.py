import numpy as np
import sys
import csv
import pandas as pd 

#reading the data
data = pd.read_csv(sys.argv[1])

#converting to the list of sentence and sentiments
x = data['review'].tolist()
y = data['sentiment'].tolist()

#converting to someting simpler for comparision
datay = np.array([0]*len(y))
for i in range(0,len(y)):
	if(y[i] == "positive"):
		datay[i] = 1

#removing bad characters
bad_chars = [';', ':', '!', '*','?','/','\"','(',')',',','.']
for i in range(0,len(x)):
	for j in bad_chars:
		x[i] = x[i].replace(j,'').lower()

#now create the dictionary of words and 
#maintain the number of words in positive and negative review
datax = {}
wc = [0,0]
for i in range(0,len(x)):
	lab = datay[i]
	words = x[i].split(" ")
	wc[lab] = wc[lab] + len(words)
	for w in words:
		if w in datax.keys():
			datax[w][lab] = datax[w][lab] + 1.0
		else:
			datax[w]= {lab: 1.0, (1-lab):1.0}
			wc[1-lab] = wc[1-lab] + 1


#now convert count to probabilities
for k in datax.keys():
	datax[k][1] = np.log(datax[k][1]/wc[1])
	datax[k][0] = np.log(datax[k][0]/wc[0])

#now the probability of documents is
p = np.array([0.0,0.0])
p[1] = np.sum(datay)/len(datay)
p[0] = 1 - p[1]
p = np.log(p)

#now we need to predict but we have to first adjust the test data
data2 = pd.read_csv(sys.argv[2])

#convert it to list as
x = data2['review'].tolist()

#convert the data as
for i in range(0,len(x)):
	for j in bad_chars:
		x[i] = x[i].replace(j,'').lower()

#now make prediction file as
y = np.array([[0]]*len(x))

#now go for prediction as
for i in range(0,len(x)):
	temp = x[i].split(" ")
	po = p[1]
	ng = p[0]
	for w in temp:
		if w in datax.keys():
			po = po + datax[w][0]
			ng = ng + datax[w][1]
	if(po < ng):
		y[i][0] = 1;

#now write to external file
np.savetxt(sys.argv[3],y)
