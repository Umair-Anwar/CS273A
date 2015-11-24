# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

from numpy.linalg import inv, pinv
import time

mu = 0
sigma = 1
# alpha = 0.3
dim = 9
# N = 1000

def genData(alpha,N,mu1,mu2,train):
	if train:
		mu1 = Series(np.random.normal(mu, sigma,dim))
		mu2 = Series(np.random.normal(mu, sigma,dim))          
	data1 = DataFrame(np.random.normal(mu,alpha*sigma,(N,dim)))+mu1
	data2 = DataFrame(np.random.normal(mu,alpha*sigma,(N,dim)))+mu2
	return data1, data2, mu1, mu2

def genOCRData(alpha,N,mu1,mu2,train):
	k = np.random.randint(2,8)
	if train:
		mu1 = Series(np.zeros(dim-k)).append(Series(np.ones(k)))
		mu1.index = range(len(mu1))
		np.random.shuffle(mu1)
		mu1.index = range(len(mu1))
		mu2 = Series(np.zeros(dim-k)).append(Series(np.ones(k)))
		mu2.index = range(len(mu2))
		np.random.shuffle(mu2)
		mu2.index = range(len(mu2))
	data1 = DataFrame(np.random.normal(mu,alpha*sigma,(N,dim)))+mu1
	data2 = DataFrame(np.random.normal(mu,alpha*sigma,(N,dim)))+mu2
	return data1, data2, mu1, mu2



def Fisher(data1,data2):
	m1 = data1.mean()
	m2 = data2.mean()
	# Sw = data1.apply(lambda x: DataFrame(x-m1).dot(DataFrame((x-m1)).transpose(),axis=1)) + \
	# 	data2.apply(lambda x: DataFrame(x-m2).dot(DataFrame((x-m2)).transpose(),axis=1))

	Sw = np.zeros((dim,dim))
	for i in range(len(data1)):
		Sw += DataFrame(data1.ix[i]-m1).dot(DataFrame(data1.ix[i]-m1).transpose())
		Sw += DataFrame(data2.ix[i]-m2).dot(DataFrame(data2.ix[i]-m2).transpose())

	Sw_inv = inv(Sw)
	w = Sw_inv.dot(m2-m1)
	b = -w.dot((m1+m2)/2)

	return np.array([np.append(w,b)]).T

def Perceptron(data1,data2,iteration):
	# using linear regression to generate the initial w0
	X = pd.concat([data1,data2])
	X[9] = 1
	X.index = range(len(X))
	y = pd.concat([DataFrame(np.ones(len(data1))),-DataFrame(np.ones(len(data2)))])
	y.index = range(len(y))
	X_pinv = pinv(X)
	w0 = X_pinv.dot(y)

	# if dim == 2:
	# 	plt.scatter(data1[0],data1[1],c='blue',alpha=0.5)
	# 	plt.scatter(data2[0],data2[1],c='red',alpha=0.5)
	# 	x1 = np.arange(X[0].min(),X[0].max(),0.1)
	# 	x2 = -(w0[0]/w0[1])*x1-w0[2]/w0[1]
	# 	plt.plot(x1,x2)


	# using perceptron learning algorithm
	# iteration = 100
	w = w0
	for i in range(iteration):
		a = np.random.randint(len(X))
		if w.T.dot(X.ix[a])*y.ix[a][0]<0:
			w = w + DataFrame(y.ix[a][0]*X.ix[a]).values

	# if dim == 2:
	# 	plt.scatter(data1[0],data1[1],c='blue',alpha=0.5)
	# 	plt.scatter(data2[0],data2[1],c='red',alpha=0.5)
	# 	x1 = np.arange(X[0].min(),X[0].max(),0.1)
	# 	x2 = -(w[0]/w[1])*x1-w[2]/w[1]
	# 	plt.plot(x1,x2)

	return w

def Test(data1,data2,w):
	data1[9] = 1
	data2[9] = 1

	label1 = data1.apply(lambda x: x.dot(w)[0]<=0,axis=1)
	label2 = data2.apply(lambda x: x.dot(w)[0]>0,axis=1)
	accuracy = (len(label1[label1==True])+len(label2[label2==True]))*1.0/(len(label1)+len(label2))
	# if accuracy < 0.99:
	# 	print accuracy
	# 	print "label1:",label1.value_counts()[False]*1.0/len(label1), "label2:",label2.value_counts()[False]*1.0/len(label2)
	# 	print label1,label2
	# print "accuracy: " + str(accuracy)
	return accuracy

def FisherTest(testNum,alpha_range,N_range,data_func,file_Name):
	fisher = DataFrame(columns=['function','alpha','N','testNum','train_time','accuracy_mean','accuracy_std']) # DataFrame used to store fisher's test results
	print "Train and Test Fisher for different alpha and N ..."
	for alpha in alpha_range:
	    for N in N_range:
	        accuracy = []
	        d1,d2,m1,m2 = data_func(alpha,N,-1,-1,train=True) # generate train data
	        start = time.time()
	        w = Fisher(d1,d2) # generate train data
	        end = time.time()
	        elapsed = end - start # calculate train time
	        for i in range(testNum):
	            d1,d2,m1,m2 = data_func(alpha,N,m1,m2,train=False) # generate test data
	            a = Test(d1,d2,w) # perform test
	            # if a < 0.9:
	            # 	print "*****"
	            	# print d1.mean().dot(w),d2.mean().dot(w)
	            accuracy.append(a)
	        # print "Fisher"+", alpha: "+str(alpha)+", N: "+str(N)+", time: "+str(elapsed)+", acc_mean: "+str(np.mean(accuracy))+", acc_std: "+str(np.std(accuracy))
	        # print accuracy
	        fisher.loc[len(fisher)] = ['Fisher',alpha,N,testNum,elapsed,np.mean(accuracy),np.std(accuracy)] # add result to the DataFrame
	print "Writing to output files ..."
	fisher.to_csv("./test_results/"+file_Name+".csv", index=False)

def PLATest(testNum,alpha,N,iter_range,testN_range,data_func,file_Name):
	perceptron = DataFrame(columns=['function','alpha','N','iteration','testN','train_time','accuracy_mean','accuracy_std']) # DataFrame used to store PLA's test results
	alpha = 0.4
	N = 500
	print "Train and Test PLA for different alpha and N ..."
	for iteration in iter_range:
	    for testN in testN_range:
	        train_d1, train_d2, mu1, mu2 = data_func(alpha,N,-1,-1,train=True) # generate train data
	       	start = time.time()
	        w = Perceptron(train_d1,train_d2,iteration) # train
	        end = time.time()
	        elapsed = end - start
	        accuracy = []
	        for i in range(testNum):
	            test_d1,test_d2,mu1,mu2 = data_func(alpha,testN,mu1,mu2,train=False) # generate test data
	            a = Test(test_d1,test_d2,w)	# test
	            accuracy.append(a)
	        # print "PLA"+", iter: "+str(iteration)+", testN: "+str(testN)+", time: "+str(mean(times))+", acc_mean: "+str(np.mean(accuracy))+", acc_std: "+str(np.std(accuracy))
	        perceptron.loc[len(perceptron)] = ['PLA',alpha,N,iteration,testN,elapsed,np.mean(accuracy),np.std(accuracy)]
	print "Writing to output files ..."
	perceptron.to_csv("./test_results/"+file_Name+".csv",index=False)


if __name__ == "__main__":

	print "========= Perceptron learning ========="
	print "=========== 1(a) Generate Data ============"
	print "generating sample data ..."
	data1,data2,mu1,mu2 = genData(0.4,400,-1,-1,train=True) # example of generate sample data
	data1.to_csv("./sample_data/problem1_data1.csv",index=False) # writing sample data to output file
	data2.to_csv("./sample_data/problem1_data2.csv",index=False)

	print "============= 1(b) Fisher’s linear discriminant and Evaluation ============"
	FisherTest(testNum=100,alpha_range=np.arange(0.1,1,0.1),N_range=range(100,1100,100),data_func=genData,file_Name="Fisher")

	print "============= 1(c) Perceptron Learning Algorithm ============="
	PLATest(testNum=100,alpha=0.4,N=500,iter_range=np.arange(100,1100,100),testN_range=range(100,1100,100),data_func=genData,file_Name="PLA")

	print "=========== Toy OCR ============="
	print "============= 2(a) Generate Data =============="
	data1,data2,mu1,mu2 = genOCRData(0.4,500,-1,-1,train=True) # example of generate sample data
	data1.to_csv("./sample_data/problem2_data1.csv",index=False) # writing sample data to output file
	data2.to_csv("./sample_data/problem2_data2.csv",index=False)

	print "============= 2(b) Fisher’s linear discriminant and Evaluation on OCR ============"
	FisherTest(testNum=100,alpha_range=np.arange(0.1,1,0.1),N_range=range(100,1100,100),data_func=genOCRData,file_Name="OCR_Fisher")

	print "============= 2(c) Perceptron Learning Algorithm on OCR ============="
	PLATest(testNum=100,alpha=0.4,N=500,iter_range=np.arange(100,1100,100),testN_range=range(100,1100,100),data_func=genOCRData,file_Name="OCR_PLA")

















