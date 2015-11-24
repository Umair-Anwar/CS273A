import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
from math import e
import math
from sympy import *

dim = 2
numCluster = 10
mu = 0
sigma = 10
N = 100
alpha = 0.05

def genData(alpha,numCluster,N):
	mus = []
	for i in range(numCluster):
		mus.append(np.random.normal(mu, sigma,dim))
	for i in range(numCluster):
		if i==0:
			datas = np.random.normal(mu,alpha*sigma,(N,dim))+mus[i]
			ys = i*np.ones(N)
		else:
			datas = np.concatenate((datas,np.random.normal(mu,alpha*sigma,(N,dim))+mus[i]),axis = 0)
			ys = np.concatenate((ys,i*np.ones(N)),axis = 0)     
	return datas,ys, np.array(mus)

def Kmeans(datas,ys,mus,true_mus):
	r = np.zeros(len(ys))
	mus_old = mus
	while True:
		# r_old = r.copy()
		for i in range(len(r)):
			c = 0
			d = np.linalg.norm(datas[i]-mus[0])
			for j in range(1,len(mus)):
				if np.linalg.norm(datas[i]-mus[j]) < d:
					c = j
					d = np.linalg.norm(datas[i]-mus[j])
			r[i] = c
		for i in range(len(mus)):
			tmp = []
			for j in range(len(r)):
				if r[j] == i:
					tmp.append(datas[j])
			mus[i] = np.mean(tmp)
		if np.linalg.norm(mus-mus_old)==0:
			break
		mus_old = mus
		# if np.linalg.norm(r-r_old)==0:
		# 	break;
	# print "error:",sum(1*(r==ys))*1.0/len(r)
	# print "Kmeans: ",np.linalg.norm(mus-true_mus)*1.0/np.linalg.norm(true_mus)
	return np.linalg.norm(mus-true_mus)*1.0/np.linalg.norm(true_mus)

def Guassin(X,mu,sigma):
	return (1.0/(2*math.pi)**(len(mu)/2)) * ((1.0/np.linalg.norm(sigma))**(1/2)) * e**((-1/2)*np.dot(np.dot((X-mu).T,np.linalg.pinv(sigma)),X-mu))

def Gamma(xs,mus,sigmas,pis,n,k):
	N = xs.shape[0]
	K = numCluster
	tmp = 0
	for j in range(K):
		tmp += pis[j]*Guassin(xs[n],mus[j],sigmas[j])
	return pis[k]*Guassin(xs[n],mus[k],sigmas[k])*1.0/tmp

def EM(xs,ys,mus,sigmas,pis,true_mus):
	N = len(ys)
	K = numCluster
	N_ks = np.zeros(K)
	mus_old  = mus
	i = 0
	while True:
		# E step
		for k in range(K):
			N_k = 0
			for n in range(N):
				N_k+= Gamma(xs,mus,sigmas,pis,n,k)
			N_ks[k] = N_k

		# M step
		# mu
		for k in range(K):
			tmp = 0
			for n in range(N):
				tmp += Gamma(xs,mus,sigmas,pis,n,k)*xs[n]
			mus[k] = tmp*1.0/N_ks[k]

		# sigma
		for k in range(K):
			tmp = 0
			for n in range(N):
				tmp += Gamma(xs,mus,sigmas,pis,n,k)*np.outer(xs[n]-mus[k],(xs[n]-mus[k]).T)
			sigmas[k] = tmp*1.0/N_ks[k]

		# pi
		for k in range(K):
			pis[k] = N_ks[k]*1.0/N

		if np.linalg.norm(mus - mus_old) < 1e-8:
			break
		mus_old = mus
		i += 1
		if i > 1e7:
			break

	# print "EM: ", np.linalg.norm(mus-true_mus)*1.0/np.linalg.norm(true_mus)
	return np.linalg.norm(mus-true_mus)*1.0/np.linalg.norm(true_mus)

def initialize(d,k,n):
	mu0 =[]
	index = []
	while True:
		i = np.random.randint(n)
		if i not in index:
			mu0.append(d[i])
			index.append(i)
		if len(mu0)== k:
			break
	mu0 = np.array(mu0)
	sigmas = np.array([np.eye(dim)] * k)
	pis = np.array([1.0/k] * k)
	return mu0,sigmas,pis






if __name__ == "__main__":


	kmeans = pd.DataFrame(columns = ["K","N","error","error_bar","time","time_bar"])
	em = pd.DataFrame(columns = ["K","N","error","error_bar","time","time_bar"])
	for k in [10,15,20]:
		for n in [100,150,200]:
			ke = []
			kt = []
			ee = []
			et = []
			for i in range(1):
				print i
				d,y,m= genData(alpha,k,n)
				mu0,sigmas,pis = initialize(d,k,n)
				a = time.time()
				e1 = Kmeans(d,y,mu0,m)
				b = time.time()	
				e2 = EM(d,y,mu0,sigmas,pis,m)
				c = time.time()
				ke.append(e1)
				kt.append(b-a)
				ee.append(e2)
				et.append(c-b)
			print "===== K: ",k,"N: ",n,"====="
			print "Kmeans", \
			"error: ",np.mean(ke)," +- ",np.sqrt(1.0/9)*np.std(ke), \
			"time: ",np.mean(kt)," +- ",np.sqrt(1.0/9)*np.std(kt)
			print "EM", \
			"error: ",np.mean(ee)," +- ",np.sqrt(1.0/9)*np.std(ee), \
			"time: ",np.mean(et)," +- ",np.sqrt(1.0/9)*np.std(et)

			kmeans.loc[len(kmeans)] = [k,n,np.mean(ke),np.sqrt(1.0/9)*np.std(ke),np.mean(kt),np.sqrt(1.0/9)*np.std(kt)]
			em.loc[len(em)] = [k,n,np.mean(ee),np.sqrt(1.0/9)*np.std(ee),np.mean(et),np.sqrt(1.0/9)*np.std(et)]
			kmeans.to_csv("./test_results/kmeans.csv")
			em.to_csv("./test_results/em.csv")

	kmeans.to_csv("./test_results/kmeans.csv")
	em.to_csv("./test_results/em.csv")








