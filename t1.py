#Name: Vipul Ramtekkar 
#Rollno: 16D110013
#Assignment 1

import numpy as np 
import math 

def user_func(x_star):
	y = 2*x_star[1]*x_star[2]*math.sin(x_star[0]) + 3*x_star[0]*math.sin(x_star[2])*math.cos(x_star[1])
	return y 

def gradient_compute(x_star):

	h = 0.001
	n = x_star.shape[0]
	e = np.eye(n)
	y = np.zeros([n,1])

	for index_star in range(len(x_star)):
		nx_star = x_star + h*e[:,index_star].reshape(n,1)
		mx_star = x_star - h*e[:,index_star].reshape(n,1)
		y[index_star, 0] = (user_func(nx_star) - user_func(mx_star))/(2*h)

	return y

def forward_diff(x_star):

	h = 0.001
	n = x_star.shape[0]
	e = np.eye(n)
	y = np.zeros([n,1])

	for index_star in range(len(x_star)):
		nx_star = x_star + h*e[:,index_star].reshape(n,1)
		mx_star = x_star
		y[index_star, 0] = (user_func(nx_star) - user_func(mx_star))/h

	return y

def analytical_grad(x_star):

	y = np.zeros([x_star.shape[0],1])
	y[0] = 2*x_star[1]*x_star[2]*math.cos(x_star[0]) + 3*math.sin(x_star[2])*math.cos(x_star[1])
	y[1] = 2*x_star[2]*math.sin(x_star[0]) - 3*x_star[0]*math.sin(x_star[2])*math.sin(x_star[1])
	y[2] = 2*x_star[1]*math.sin(x_star[0]) + 3*x_star[0]*math.cos(x_star[2])*math.cos(x_star[1])
	
	return y

x_star = np.array([[-1],[1],[1]])

print gradient_compute(x_star)
print forward_diff(x_star)
print analytical_grad(x_star)


