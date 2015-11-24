from __future__ import division
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
from math import e
import math
from sympy import *

input_dim = 100
hidden_dim = 30
output_dim = 100
alpha = 0.1
B = 100
lamb = 0.001
data_num = 10000
iteration = 20000
seed = 1234

def sigmoid(x):
	return 1/(1+e**-x)

def drange(start, stop, step):
	t = start
	r = []
	while t <= stop:
		r.append(t)
		t += step
	return r

def exactOneOneData(n_data,n_dim,seed):
	np.random.seed(seed)
	# number of datapoints
	# n_data = 15
	# number of features or dimensions
	# n_dim = 6
	p_threshold = 2 / n_dim
	p_matrix = np.random.rand(n_data , n_dim)
	np.random.rand(n_data , n_dim)
	# binarize this matrix so that values bigger than p_threshold will become 1 # and the rest become 0
	x_matrix = 1 * (p_matrix < p_threshold)
	# find rows which have exactly one element turned on
	y = 1 * (np.sum(x_matrix, 1) == 1)
	# helper function for pretty printing
	def print_named(thing_to_print , name): 
	    header_bar = '=' * (len(name) + 1) 
	    print header_bar
	    print name + ':'
	    print header_bar 
	    print thing_to_print 
	    print
	# print_named(p_matrix , 'p_matrix')
	# print_named(p_threshold , 'p_threshold')
	# print_named(x_matrix , 'x_matrix')
	# print_named(y, 'y')

	return x_matrix, y

def exactOneOn_train(data_num,input_dim,hidden_dim,output_dim,iteration,B,alpha,lamb,seed):
	x_matrix, y = exactOneOneData(data_num,input_dim,seed)
	W01 = np.random.randn(input_dim+1,hidden_dim+1)
	W12 = np.random.randn(hidden_dim+1,output_dim)
	W = np.concatenate((W01,W12.T),axis=0)
	W_old = W
	
	for i in range(iteration):
	# while True:
		mini_batch=[]
		while len(set(mini_batch))<B:
			mini_batch.append(np.random.randint(0,data_num-1))
		mini_batch = list(set(mini_batch))
		W_mini = np.zeros([input_dim+1+output_dim,hidden_dim+1])
		# print mini_batch
		for j in mini_batch:
			W01 = W[:input_dim+1]
			W12 = W[input_dim+1:].T
			A0 = np.append(x_matrix[j],1)
			Z0 = A0
			A1 = Z0.dot(W01)
			Z1 = sigmoid(A1)
			A2 = Z1.dot(W12)
			Z2 = A2
			# Z2 = np.array([1]) if Z2>0.5 else np.array([0])
			# print Z2
			# Z2 = sigmoid(A2)
			sigma2 = Z2 - y[j]
			sigma1 = sigmoid(A1)*(1-sigmoid(A1)) * sigma2.dot(W12.T) # potential bug
			# print Z1.T.shape,sigma2.shape
			W12_der = np.outer(Z1,sigma2)
			W01_der = np.outer(Z0,sigma1)
			W_der = np.concatenate((W01_der,W12_der.T),axis=0)
			W_mini += W_der
		W_mini /= B
		# print np.linalg.norm(W_mini)
		W = W - alpha/np.linalg.norm(W_mini)*(W_mini+lamb/2*W)
		# print np.linalg.norm(W-W_old)
		# if np.linalg.norm(W-W_old)<1e-3:
		# 	break
		# predict = []
		# for k in range(data_num):
		# 	A0 = np.append(x_matrix[k],1)
		# 	Z0 = A0
		# 	# print "A0",A0
		# 	A1 = Z0.dot(W[:input_dim+1])
		# 	# print "A1",A1
		# 	Z1 = sigmoid(A1)
		# 	A2 = Z1.dot(W[input_dim+1:].T)
		# 	Z2 = A2
		# 	p = 1 if Z2>0.5 else 0
		# 	# print sigmoid(Z2)
		# 	predict.append(p)

		# print predict,y
		# print "==== Train ====="
		# print "predict",sum(predict),"y",sum(y)
		# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
		# print i, "precision",sum(1*(predict==y))/data_num,"gradient",np.linalg.norm(W_mini),"W",np.linalg.norm(W-W_old)
		W_old = W

		# W = W - alpha*(W_mini)
	# print Z1.shape,W[input_dim+1:].shape
	# print "W",W
	predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		p = 1 if Z2>0.5 else 0
		# print sigmoid(Z2)
		predict.append(p)

	# print predict,y
	# print "==== Train ====="
	# print "predict",sum(predict),"y",sum(y)
	# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
	# print "precision",sum(1*(predict==y))/data_num

	return W, sum(1*(predict==y))/data_num

def exactOneOn_test(W,data_num,hidden_dim,input_dim,seed,m1=0,m2=0):
	x_matrix, y = exactOneOneData(data_num,input_dim,seed)
	predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		p = 1 if Z2>0.5 else 0
		# print sigmoid(Z2)
		predict.append(p)

	# print "==== Test ===="
	# print "predict",sum(predict),"y",sum(y)
	# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
	# print "precision",sum(1*(predict==y))/data_num
	return sum(1*(predict==y))/data_num

def autocoderData(n_data,n_dim_limited,n_dim_full,seed):
	np.random.seed(seed)
	# n_dim_full = 100
	# n_dim_limited = 30
	# n_data = 10000
	eigenvals_big = np.random.randn(n_dim_limited) + 3
	eigenvals_small = np.abs(np.random.randn(n_dim_full - n_dim_limited)) * .1
	eigenvals = np.concatenate([eigenvals_big , eigenvals_small])
	# print eigenvals
	diag = np.diag(eigenvals)
	q, r = np.linalg.qr(np.random.randn(n_dim_full , n_dim_full))
	cov_mat = q.dot(diag).dot(q.T)
	mu = np.zeros(n_dim_full)
	x = np.random.multivariate_normal(mu, cov_mat , n_data)
	return x

# =============== Exactly one on ====================	

# ================== Autocoder ===================
def AutoCoder_train(data_num,input_dim,hidden_dim,output_dim,iteration,B,alpha,lamb,seed):
	x_matrix = autocoderData(data_num,hidden_dim,input_dim,seed)
	W01 = np.random.randn(input_dim+1,hidden_dim+1)
	W12 = np.random.randn(hidden_dim+1,output_dim)
	W = np.concatenate((W01,W12.T),axis=0)
	W_old = W
	
	for i in range(iteration):
		
	# while True:
		mini_batch=[]
		while len(set(mini_batch))<B:
			mini_batch.append(np.random.randint(0,data_num-1))
		mini_batch = list(set(mini_batch))
		W_mini = np.zeros([input_dim+1+output_dim,hidden_dim+1])
		# print mini_batch
		for j in mini_batch:
			W01 = W[:input_dim+1]
			W12 = W[input_dim+1:].T
			A0 = np.append(x_matrix[j],1)
			Z0 = A0
			A1 = Z0.dot(W01)
			Z1 = sigmoid(A1)
			A2 = Z1.dot(W12)
			Z2 = A2
			# Z2 = np.array([1]) if Z2>0.5 else np.array([0])
			# print Z2
			# Z2 = sigmoid(A2)
			sigma2 = Z2 - x_matrix[j]
			sigma1 = sigmoid(A1)*(1-sigmoid(A1)) * sigma2.dot(W12.T) # potential bug
			# print Z1.T.shape,sigma2.shape
			W12_der = np.outer(Z1,sigma2)
			W01_der = np.outer(Z0,sigma1)
			W_der = np.concatenate((W01_der,W12_der.T),axis=0)
			W_mini += W_der
		W_mini /= B
		# print np.linalg.norm(W_mini)
		# W = W - alpha*np.linalg.norm(W_mini)*(W_mini+lamb/2*W)
		W = W - alpha*(W_mini+lamb/2*W)
		# print np.linalg.norm(W-W_old)
		# if i%1000==0:
		# 	error = 0
		# 	baseline = 0
		# 	predict = []
		# 	for k in range(data_num):
		# 		A0 = np.append(x_matrix[k],1)
		# 		Z0 = A0
		# 		# print "A0",A0
		# 		A1 = Z0.dot(W[:input_dim+1])
		# 		# print "A1",A1
		# 		Z1 = sigmoid(A1)
		# 		A2 = Z1.dot(W[input_dim+1:].T)
		# 		Z2 = A2
		# 		# print sigmoid(Z2)
		# 		predict.append(Z2)
		# 		error += np.linalg.norm(Z2-x_matrix[k])
		# 		baseline += np.linalg.norm(x_matrix[k])

		# 	# print predict,y
		# 	print i,"error", error/data_num,"gradient",np.linalg.norm(W_mini)
		W_old = W

		# W = W - alpha*(W_mini)
		# print Z1.shape,W[input_dim+1:].shape
		# print "W",W
	error = 0
	baseline = 0
	# predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		# print sigmoid(Z2)
		# predict.append(Z2)
		error += np.linalg.norm(Z2-x_matrix[i])
		baseline += np.linalg.norm(x_matrix[i])

	# print predict,y
	# print "==== Train ====="
	# print "baseline", baseline/data_num
	# print "error", error/data_num

	return W, error/data_num

def AutoCoder_test(W,data_num,hidden_dim,input_dim,seed,m1=0,m2=0):

	x_matrix = autocoderData(data_num,hidden_dim,input_dim,seed)
	error = 0
	baseline = 0
	# predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		# print sigmoid(Z2)
		# predict.append(Z2)
		error += np.linalg.norm(Z2-x_matrix[i])
		baseline += np.linalg.norm(x_matrix[i])

	# print "==== Test ===="
	# print "baseline", baseline/data_num
	# print "error", error/data_num
	return error/data_num

def imageData(a,N,dim,mu1,mu2,train):
	dim = 35
	k = 30
	if train:
		mu1 = Series(np.zeros(dim-k)).append(Series(np.ones(k)))
		mu1.index = range(len(mu1))
		np.random.shuffle(mu1)
		mu1.index = range(len(mu1))
		mu2 = Series(np.zeros(dim-k)).append(Series(np.ones(k)))
		mu2.index = range(len(mu2))
		np.random.shuffle(mu2)
		mu2.index = range(len(mu2))
	data1 = DataFrame(np.random.normal(0,a*1,(N/2,dim)))+mu1
	data2 = DataFrame(np.random.normal(0,a*1,(N/2,dim)))+mu2
	y = np.append(np.ones(N/2),np.zeros(N/2))
	return np.append(data1,data2,0), y, mu1, mu2

def image_train(data_num,input_dim,hidden_dim,output_dim,iteration,B,alpha,lamb,seed):
	x_matrix, y, m1, m2 = imageData(0.3, data_num,input_dim,-1,-1,train = True)
	W01 = np.random.randn(input_dim+1,hidden_dim+1)
	W12 = np.random.randn(hidden_dim+1,output_dim)
	# W01 = np.zeros([input_dim+1,hidden_dim+1])
	# W12 = np.zeros([hidden_dim+1,output_dim])
	W = np.concatenate((W01,W12.T),axis=0)
	W_old = W
	
	for i in range(iteration):	
	# while True:
		mini_batch=[]
		while len(set(mini_batch))<B:
			mini_batch.append(np.random.randint(0,data_num-1))
		mini_batch = list(set(mini_batch))
		W_mini = np.zeros([input_dim+1+output_dim,hidden_dim+1])
		# print mini_batch
		for j in mini_batch:
			W01 = W[:input_dim+1]
			W12 = W[input_dim+1:].T
			A0 = np.append(x_matrix[j],1)
			Z0 = A0
			A1 = Z0.dot(W01)
			Z1 = sigmoid(A1)
			A2 = Z1.dot(W12)
			Z2 = A2
			# Z2 = np.array([1]) if Z2>0.5 else np.array([0])
			# print Z2
			# Z2 = sigmoid(A2)
			sigma2 = Z2 - y[j]
			sigma1 = sigmoid(A1)*(1-sigmoid(A1)) * sigma2.dot(W12.T) # potential bug
			# print Z1.T.shape,sigma2.shape
			W12_der = np.outer(Z1,sigma2)
			W01_der = np.outer(Z0,sigma1)
			W_der = np.concatenate((W01_der,W12_der.T),axis=0)
			W_mini += W_der
		W_mini /= B
		# print np.linalg.norm(W_mini)
		# W = W - alpha/np.linalg.norm(W_mini)*(W_mini+lamb/2*W)
		W = W - alpha*(W_mini+lamb/2*W)
		# predict = []
		# for k in range(data_num):
		# 	A0 = np.append(x_matrix[k],1)
		# 	Z0 = A0
		# 	# print "A0",A0
		# 	A1 = Z0.dot(W[:input_dim+1])
		# 	# print "A1",A1
		# 	Z1 = sigmoid(A1)
		# 	A2 = Z1.dot(W[input_dim+1:].T)
		# 	Z2 = A2
		# 	p = 1 if Z2>0.5 else 0
		# 	# print sigmoid(Z2)
		# 	predict.append(p)

		# print predict,y
		# print "==== Train ====="
		# print "predict",sum(predict),"y",sum(y)
		# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
		# print i, "precision",sum(1*(predict==y))/data_num,"gradient",np.linalg.norm(W_mini),"W",np.linalg.norm(W-W_old)
		W_old = W

		# W = W - alpha*(W_mini)
		# print Z1.shape,W[input_dim+1:].shape
		# print "W",W
	predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		# print Z2
		p = 1 if Z2>0.5 else 0
		# print sigmoid(Z2)
		predict.append(p)

	# print predict,y
	# print "==== Train ====="
	# print "predict",sum(predict),"y",sum(y)
	# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
	# print "precision",sum(1*(predict==y))/data_num
	return W, sum(1*(predict==y))/data_num,m1,m2

def image_test(W,data_num,hidden_dim,input_dim,seed,m1,m2):
	x_matrix, y, m1, m2 = imageData(0.3, data_num,input_dim,m1,m2,train = False)
	predict = []
	for i in range(data_num):
		A0 = np.append(x_matrix[i],1)
		Z0 = A0
		# print "A0",A0
		A1 = Z0.dot(W[:input_dim+1])
		# print "A1",A1
		Z1 = sigmoid(A1)
		A2 = Z1.dot(W[input_dim+1:].T)
		Z2 = A2
		p = 1 if Z2>0.5 else 0
		# print sigmoid(Z2)
		predict.append(p)

	# print predict,y
	# print y.shape
	# print "==== Test ====="
	# print "predict",sum(predict),"y",sum(y)
	# print "baseline",sum(1*(np.zeros(data_num)==y))/data_num
	# print "precision",sum(1*(predict==y))/data_num
	return sum(1*(predict==y))/data_num



def Test(train, test,data_num,input_dim,hidden_dim,output_dim,iteration_range,B,alpha_range,lamb_range,seed):
	result = DataFrame(columns = ['ANN','data_num','input_dim','hidden_dim','output_dim','iteration','B','alpha','lamb','seed','train_accuracy','accuracy','error_bar'])
	for a in alpha_range:
		for i in iteration_range:
			for l in lamb_range:
				# print a, i, l
				test_ps = []
				for j in range(10):
					print j
					if train.__name__=="image_train":
						w, train_p,m1,m2 = train(data_num,input_dim,hidden_dim,output_dim,i,B,a,l,seed)
					else:
						w, train_p = train(data_num,input_dim,hidden_dim,output_dim,i,B,a,l,seed)
					s = np.random.randint(0,1000)
					test_p = test(w,data_num,hidden_dim,input_dim,s,m1,m2)
					test_ps.append(test_p)
				r = [train.__name__,data_num,input_dim,hidden_dim,output_dim,i,B,a,l,s,train_p,np.mean(test_ps),np.sqrt(1.0/(10-1))*np.std(test_ps)]
				print train.__name__, "alpha:",a, "iteration:",i, "lambda:", l, "train_accuracy:",train_p,"accuracy:", np.mean(test_ps),"+=",np.sqrt(1.0/(10-1))*np.std(test_ps)
				result.loc[len(result)] = r
	result.to_csv(train.__name__+'.csv')

def hinton(matrix):
    ax = plt.gca()

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


if __name__ == "__main__":


	input_dim = 5
	hidden_dim = 7
	output_dim = 1
	alpha = 0.1
	B = 100
	lamb = 0.001
	data_num = 1000
	iteration = 3000
	seed = 1111
	alpha_range = drange(0.05,0.1,0.01)
	lamb_range = drange(0.005,0.01,0.001)
	iteration_range = drange(1000,3000,1000)
	# alpha_range = [0.1]
	# lamb_range = [0.001]
	# iteration_range = [100]


	Test(exactOneOn_train, exactOneOn_test,data_num,input_dim,hidden_dim,output_dim,iteration_range,B,alpha_range,lamb_range,seed)

	input_dim = 100
	hidden_dim = 30
	output_dim = 100
	alpha = 0.1
	B = 100
	lamb = 0.001
	data_num = 10000
	iteration = 5000
	seed = 2222
	alpha_range = drange(0.05,0.1,0.01)
	lamb_range = drange(0.005,0.01,0.001)
	iteration_range = drange(1000,5000,1000)
	alpha_range = [0.1]
	lamb_range = [0.001]
	iteration_range = [1000]
	Test(AutoCoder_train, AutoCoder_test,data_num,input_dim,hidden_dim,output_dim,iteration_range,B,alpha_range,lamb_range,seed)


# ==============================  Image Classification ===========================

	input_dim = 35
	hidden_dim = 50
	output_dim = 1
	alpha_range = [0.01,0.1]
	B = 100
	lamb_range = [0.001,0.01]
	data_num = 1000
	iteration_range = [500,1000]
	seed = 0
	Test(image_train, image_test,data_num,input_dim,hidden_dim,output_dim,iteration_range,B,alpha_range,lamb_range,seed)

	

	



