#Name: Vipul Ramtekkar 
#Rollno: 16D110013
#Assignment 2

import numpy as np 
import math 

def function(x):
	return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def grad_analytical(x):
	return np.array([[200*(x[1]-x[0]**2)*(-2*x[1]) - 2*x[0],200*(x[1]-x[0]**2)]])

def grad_hessian(x):
	return np.array([[200*(-2*x[1])-400, 800*x[1]*x[0]],[200, -400*x[0]]])

x_init_guess = np.array([[2],[2]])
N = 100 
tol = 1e-8
x1 = []
x2 = []
iteration_num = []

val = []
for i in range(1,N):

	val.append(function(x_init_guess))
	x1.append(x_init_guess[0])
	x2.append(x_init_guess[1])
	iteration_num.append(i)
	p = np.linalg.solve(grad_hessian(x_init_guess),-grad_analytical(x_init_guess))
	'''
	x_init_guess = x_init_guess  + p 

	if grad_hessian(x_init_guess) < tol:
		break

import matplotlib.pyplot as plt 

plt.plot(x1,x2,iteration_num)
plt.plot(val,iteration_num)
plt.show()'''