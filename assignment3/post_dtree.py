# now need to do one hot encoding of data as follows
# first import the libraries
import numpy as np
from numpy import genfromtxt
import sys
import queue
import matplotlib.pyplot as plt
'''
we have three parameters
1. train data
2. test data
3. validation data
now read one by one
'''

# the train data
# delete the last column in data also delete the first row
data = genfromtxt(sys.argv[1], dtype=str, delimiter=",")
data = np.delete(data, 0, axis=0)
datay = data[:,len(data[0])-1:len(data[0])].astype(np.int)

# the validation data
valid = genfromtxt(sys.argv[2], dtype=str, delimiter=",")
valid = np.delete(valid, 0, axis=0)
validy = valid[:, len(valid[0]) - 1:].astype(np.int)
valid = np.delete(valid, len(valid[0]) - 1, axis=1)

# the test data
test = genfromtxt(sys.argv[3], dtype=str, delimiter=",")
test = np.delete(test, 0, axis=0)
test = np.delete(test, len(test[0]) - 1, axis=1)

# print the size of data
print(data.shape, valid.shape, test.shape)

# now need to do one hot encoding as follows
values = [" con, z1",
          " Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked",
          " con, z2",
          " Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",
          " con, z3",
          " Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse",
          " Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces",
          " Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",
          " White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",
          " Female, Male",
          " con, z4",
          " con, z5",
          " con, z6",
          " United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands"]

# now convert above one to 2d array
'''
#created a variable t and then update it
#converting it to suitable 2d array
'''
##just find the splitting condtion
# we have spited in sorting order
t = [];
for i in range(0, len(values)):
    t = t + [np.sort(values[i].split(","))]
values = np.array(t)



# lets just build the all in one hot encoder
# we are also endcoding continuous variable with medain condtion
def onehot(datax, a):
    '''
	based on this encoding give ranking for splittings
	the a specifies which attributes are available for splitting
	we also require a array to know the boundaries of the attributes
	'''

    '''
	the following segment finds the boundaries for continious variables
	it firt sorts them and take median then
	'''
    # need to redefine data according to a
    # ----------REDEFINING DATA ACCORDING TO A------------
    # defining the v as follows
    #   v: the possible values of attributs a
    #	b: the boundary of attributes
    #	the boundary of ith variable will be from ith to (i+1)th index value of b
    #	now datax will be choose manner of a
    #	datay is inbuilt in datax
    v = []
    b = [0]
    datay = datax[:, len(datax[0]) - 1:len(datax[0])].astype(np.int)
    data = datax[:, 0:1]
    for i in range(0, len(a)):
        data = np.append(data, datax[:, a[i]:a[i] + 1], axis=1)
        if (values[a[i]][0] == ' con'):
            z = datax[:, a[i]:a[i] + 1]
            z = np.sort(z, axis=0)
            v = v + [np.array([' con', values[a[i]][1], z[int(len(data) / 2)][0]])]
        else:
            v = v + [values[a[i]]]
        b = b + [b[-1] + len(v[-1])]
    data = np.delete(data, 0, axis=1)
    # ----------FINDING MEDIAN------------
    # for i in range(0,len(v)):
    # 	if (v[i][0] == ' con'):
    # 		z = (data[:,i:i+1]).astype(np.int64)
    # 		z = np.sort(z, axis = 0)
    # 		print(str(z[13500][0]))
    # 		v[i][1] = str(z[13500][0])
    # 		print(v[i][1])

    '''
	from here do one hot encoding for test data
	for continious variable if x > split_value: 0 1
	else: 1 0
	for discrete variables just do one hot encoding
	'''
    # the temprory variable t
    t = np.array([[0]]*len(data))

    # -----------ONE HOT ENCODING--------------
    for i in range(0, len(v)):
        # ------Continious variable---------------
        if (v[i][0] == " con"):
            # the condition for continious variable
            # it will be nx2 matrix
            t1 = np.array([[0] * 2] * len(data))
            # t2 converts str data to int
            t2 = (data[:, i:i + 1]).astype(np.int)
            o = int(v[i][2])
            # do the splitting about o
            for j in range(0, len(data)):
                if (t2[j][0] <= o):
                    t1[j][0] = 1
                else:
                    t1[j][1] = 1
            t = np.append(t, t1, axis=1)

        # ---------Discrete Variable------------
        else:
            # else in continious case do one hot encoding
            t1 = np.array([[0] * len(v[i])] * len(data))
            for j in range(0, len(data)):
                t1[j][np.where(v[i] == data[j][i])[0][0]] = 1
            t = np.append(t, t1, axis=1)
    t = np.delete(t, 0, axis=1)

    # -------------FINDING BEST ATTRIBUTE TO SPLIT--------------
    # the number of examples
    # this sector is really messy lol
    # i defined many varaibles here as
    # te -> denotes the total example with particular positive feature value
    #     for example te[i] gives number of example with ith feature true
    # r will be return to get further answers
    te = np.sum(t, axis=0)
    r = te
    # lets avoid number of exceptions by having atleast one example as this
    te1 = te
    te1[te1 == 0] = 1

    # tp denotes that given example has positive in ith feature then what is number of
    # positive outcome
    tp = (np.sum(t * datay, axis=0))

    # ig will find entropy of every attribute
    # ig is important variable and will help us for splitting
    ig = np.array([0.0] * len(data[0]))

    # igf finds entropy of all features
    igf = tp / te1

    # now te[i] gives fraction of examples with ith feature positive
    # te[i] = Si/S
    te = te / 30000

    # now update ig with sum of all iths feature positive ness
    # ap = summation over all fp such that f is corresponding to a
    for i in range(0, len(ig)):
        ig[i] = np.sum(tp[b[i]:b[i + 1]]) / 30000

    # making all p+ = 0.0001 which were 0 to avoid warning in future
    igf[igf == 0.0] = 0.0001
    igf[igf == 1.0] = 0.9999
    tp = tp / te1
    # the entropy formula
    # entropy(f) = -fp*log(fp) -(1-fp)*log(1-fp)
    igf = -1 * igf * (np.log(igf)) - (1 - igf) * (np.log(1 - igf))

    # similar things for ig
    ig[ig == 0.0] = 0.0001
    ig[ig == 1.0] = 0.9999
    ig = -1 * ig * (np.log(ig)) - (1 - ig) * (np.log(1 - ig))

    # multiplying feature entropies with their fraction
    igf = igf * te
    # and now information gain formula
    for i in range(0, len(ig)):
        ig[i] = ig[i] - np.sum(igf[b[i]:b[i + 1]])

    # now find the index of element in the values
    g = np.argmax(ig)
    p = v[g][1]

    # just run for loop as followa
    for i in range(0, len(values)):
        if (values[i][1] == p):
            break

    # return the feature to split and number of elements as
    # lets check compatibility of answer
    if (ig[g] == -0.0001 * np.log(0.0001) - 0.9999 * np.log(0.9999)):
        # check if all answer are true or false
        return (datay[0][0], 'leaf')
    # else return all things
    else:
        # i need to check the pruning also that will need the te
        return (i, r[b[g]:b[g + 1]], tp[b[g]:b[g + 1]])


z = np.arange(0, 14)


# now lets define the tree nood and root and try to make one inference from it
# we will require the split value and right and left child

# -------------------------THE DECISION TREE---------------------------
class node:

    # the initiator as
    def __init__(self, k):
        self.key = k
        self.value = None
        self.children = []
        self.out = 0
        self.cont = False


# #now make class of decision tree
# #this will contain root and the depth till which the tree is
class dtree:
    # the intiator as follows
    def __init__(self, k):
        # the n stores number of nodes
        self.n = 1
        self.root = node(k)


# just write function to make inference
# the input will be string array
# the find function need not to be in class of dtree
def find(x, no):
    if (no.key == -1):
        return (no.out)
    elif (no.cont):
        if ((int)(x[no.key]) <= no.value):
            return (find(x, no.children[0]))
        else:
            return (find(x, no.children[1]))
    else:

        return (find(x, no.children[np.where(values[no.key] == x[no.key])[0][0]]))


# lests write a functiuon to check the splitting condition as
def check_split(n):
    # define temporary array as
    z = 0
    # just use find function as
    for i in range(0, len(validy)):
        if (validy[i][0] == find(valid[i], n)):
            z = z + 1
    return (z / len(validy))


# i think i will just write code to buid the tree as function
# n: is the maximum number of nodes
def build_dtree(data, n):
    # lets define a queue and work on it
    q = queue.Queue(maxsize=10000)

    # first of all define tree as
    dt = dtree(-1)

    # define a variable which keeps trak of number of nodes added to tree
    k = 1
    # we will add tuple inside the queue with 3 fiels
    # feild1 = node
    # field2 = data start and end points
    # feild4 = the attributes
    # now insert dt.root,0,30000,np.arange(0,14)
    q.put((dt.root, 0, 27000, np.arange(0, 14)))

    # define current acuracy as acc
    acc = 0.0

    # run while loop as
    while ((not q.empty()) and k < n):
        # start poping the queue
        t = q.get()
        print(k)
        # if a is not empty then do this
        if (len(t[3]) != 0 and t[1] < t[2] and t[2] < 27001):
            # run the code as
            # the s is splitting atribute
            z = onehot(data[t[1]:t[2] + 1, :], t[3])

            # see for leaf condition
            if (type(z[1]) == str and z[1] == 'leaf'):
                # just add out as
                t[0].out = z[0]

            # else do all code
            else:
                s = z[0]
                r = z[1]
                a = t[3][t[3] != s]
                t[0].key = s
                k = k + 1
                # now sort about it
                data[t[1]:t[2] + 1] = data[t[1]:t[2] + 1][data[t[1]:t[2] + 1, s].argsort()]

                # check whether it is continious split or what
                if (values[s][0] == ' con'):

                    # find the piovet as
                    i = int((t[1] + t[2]) / 2)

                    # now change node values
                    t[0].value = int(data[i][s])
                    t[0].cont = True
                    t[0].children = [node(-1), node(-1)]
                    
                # this will be executed if split is good
                # now just add children to node as
                    if (t[1] < i):
                        q.put((t[0].children[0], t[1], i, a))
                        
                    if (t[2] > 1 + i):
                        q.put((t[0].children[1], i + 1, t[2], a))
                        

                # now work on else condition as following
                else:
                    # given splits are r now just add element to queue
                    # we will need piovet as
                    p = t[1]
                    for j in range(0, len(values[s])):
                        c = node(-1)
                        t[0].children = t[0].children + [c]
                        if (z[2][j] > 0.5):
                            t[0].children[j].out = 1
                        else:
                            t[0].children[j].out = 0

                    for j in range(0, len(values[s])):
                        c = t[0].children[j]
                        if (p < p + r[j]):
                            q.put((c, p, p + r[j], a))
                            p = p + r[j]
            # else dont split the node and make it leaf

        # else we have to work on leaf nodes as follows
        else:
            # count the positive instances as
            s = np.sum(data[t[1]:t[2] + 1, len(data[0]) - 1].astype(np.int)) / (t[2] - t[1])
            # check conditions on s
            if (s >= 0.5):
                t[0].out = 1
            else:
                t[0].out = 0
    # now what if queue gets empty or k == q lets see
    if ((not q.empty()) and k >= n):
        # do the out for every one
        while (not q.empty()):
            # just pop one element as
            t = q.get()
            # count the positive instances as
            s = np.sum(data[t[1]:t[2] + 1, len(data[0]) - 1].astype(np.int)) / (t[2] - t[1])
            # check conditions on s
            if (s >= 0.5):
                t[0].out = 1
            else:
                t[0].out = 0

    return (dt, k)


# write function to see the tree
def show_tree(x, k, d):
    # just write recursive function as
    # the c denote no_of_nodes
    if (len(x.children) == 0):
        print(x.key, k, d)
    else:
        print(x.key, k, d, len(x.children))
        for i in range(0, len(x.children)):
            show_tree(x.children[i], x.key, d + 1)

def post_prune(n):
    acc = check_split(n)
    q = queue.Queue()
    for i in range(0,len(n.children)):
        q.put(n.children[i])
    while(not q.empty()):
        t = q.get()
        s0 = t.key
        t.key = -1
        t.out = 0
        w0 = check_split(n)
        t.out = 1
        w1 = check_split(n)
        if(w0 > acc and w0 > w1):
            t.out = 0
            t.children = []
        elif(w1 > acc and w1 > w0):
            t.out = 1
            t.children = []
        else:
            t.key = s0
            for i in range(0,len(t.children)):
                q.put(t.children[i])
    return(n)


def checkv(n):
    z = 0
    # now write to file
    q = np.array([[0]] * len(valid))
    for i in range(0, len(valid)):
        q[i][0] = find(valid[i], x[0].root)
        if (q[i][0] == validy[i][0]):
            z = z + 1 
    print(z / len(valid))
    return(z / len(valid))

def checkt(n):
    z = 0
    # now write to file
    q = np.array([[0]] * len(data))
    for i in range(0, len(data)):
        q[i][0] = find(data[i], x[0].root)
        if (q[i][0] == datay[i][0]):
            z = z + 1 
    print(z / len(data))
    return(z / len(data))

def checktf(n):
    z = 0
    # now write to file
    q = np.array([[0]] * len(data))
    for i in range(0, len(data)):
        q[i][0] = find(data[i], x[0].root)
        if (q[i][0] == datay[i][0]):
            z = z + 1 
    np.savetxt(sys.argv[4], q)

x = build_dtree(data,7371)
print(checkt(x[0].root))
checktf(x[0].root)
'''
v = np.array([0.0]*10)
t = np.array([0.0]*10)
vp = np.array([0.0]*10)
tp = np.array([0.0]*10)
for i in range(1,11):
    x = build_dtree(data,i)
    v[i-1] = checkv(x[0].root)
    t[i-1] = checkt(x[0].root)
    x[0].root = post_prune(x[0].root)
    vp[i-1] = checkv(x[0].root)
    tp[i-1] = checkt(x[0].root)
    print(i)
    
f = plt.figure()
plt.plot(np.arange(1,11),v, label = "validation data")
plt.plot(np.arange(1,11),t, label = "test data")

plt.title("acurracy without pruning")
plt.legend()
plt.xlabel("nodes") 
plt.ylabel('acuuracy')
f.savefig("1.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(np.arange(1,11),vp, label = "validation data")
plt.plot(np.arange(1,11),tp, label = "training data")

plt.title("acurracy with pruning")
plt.legend()
plt.xlabel("nodes") 
plt.ylabel('acuuracy')
f.savefig("2.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(np.arange(1,11),t, label = "without pruning")
plt.plot(np.arange(1,11),tp, label = "pruning")

plt.title("acurracy pruning vs non pruning on test data")
plt.legend()
plt.xlabel("nodes") 
plt.ylabel('acuuracy')
f.savefig("3.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(np.arange(1,11),v, label = "without pruning")
plt.plot(np.arange(1,11),vp, label = "pruning")

plt.title("acurracy pruning vs non pruning on validation data data")
plt.legend()
plt.xlabel("nodes") 
plt.ylabel('acuuracy')
f.savefig("4.pdf", bbox_inches='tight')
'''

# #lets try an example
# values = [np.array(['red','green','white','brown']),np.array(['big','small','medium']),np.array(['heavy','light','medium']),np.array(['con'])]
# y = dtree(0)
# for i in range(0,len(values)):
# 	values[i] = np.sort(values[i])
# print(values)
# y.root.children = [node(-1),node(-1),node(-1),node(-1)]
# y.root.children[2].key = 3
# y.root.children[2].cont = True
# y.root.children[2].value = 20
# y.root.children[2].children = [node(-1),node(-1)]
# y.root.children[2].children[0].out = 1
# y.root.children[2].children[1].out = 0
# y.root.children[0].out = 0
# y.root.children[1].children = [node(-1),node(-1),node(-1)]
# y.root.children[1].key = 2
# y.root.children[3].key = 1
# y.root.children[3].children = [node(-1),node(-1),node(-1)]
# y.root.children[3].children[2].out = 0
# y.root.children[3].children[1].out = 0
# y.root.children[3].children[0].children = [node(-1),node(-1),node(-1)]
# y.root.children[3].children[0].children[0].out = 0
# y.root.children[3].children[0].children[1].out = 1
# y.root.children[3].children[0].children[2].out = 1
# y.root.children[3].children[0].key = 2
# y.root.children[1].children[0].out = 1
# y.root.children[1].children[1].out = 0
# y.root.children[1].children[2].out = 0
# x = ['red','small','light','18']
# print (find(x,y.root))