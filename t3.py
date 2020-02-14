#Name: Vipul Ramtekkar 
#Rollno: 16D110013
#Assignment 3

import numpy as np 
import math 

def user_func(x_star):
	y = 100*(x_star[1] - x_star[0]**2 )**2 + (1 - x_star[0])**2
	return y 

def backtrackline(x_star,delta_x):

	alpha = 5
	rho = 0.8 
	c = 0.1

	while user_func(np.add(x_star, alpha*delta_x)) > user_func(x_star) + c*alpha*(np.transpose(analytical_grad(x_star)).dot(delta_x)):
		alpha = rho*alpha
	return alpha

def BFGS(N,tol,x_star):
	inv_B = np.identity(x_star.shape[0])
	x1 = []
	x2 = []
	iterations = []
	fx = []

	for i in range(N):
		x1.append(x_star[0])
		x2.append(x_star[1])
		iterations.append(i)
		fx.append(user_func(x_star))
		delta_x = -inv_B.dot(analytical_grad(x_star))
		t = backtrackline(x_star,delta_x)
		x_star_1 = np.add(x_star,t*delta_x)
		y = analytical_grad(x_star_1) - analytical_grad(x_star)
		s = x_star_1 - x_star
		a = (np.identity(x_star.shape[0])-(1/np.dot(np.transpose(y),s))*np.dot(s,np.transpose(y)))
		b = (np.identity(x_star.shape[0])-(1/np.dot(np.transpose(y),s))*np.dot(y,np.transpose(s)))
		inv_B = np.dot(np.dot(a,inv_B),b) + (1/np.dot(np.transpose(y),s))*(np.dot(s,np.transpose(s)))
		if user_func(x_star) - user_func(x_star_1) < tol:
			x_star_1 = x_star
			break
		x_star = x_star_1

	return x_star,x1,x2,fx,iterations

def DFP(N,tol,x_star):
	inv_B = np.identity(x_star.shape[0])
	x1 = []
	x2 = []
	iterations = []
	fx = []
	for i in range(N):
		x1.append(x_star[0])
		x2.append(x_star[1])
		iterations.append(i)
		fx.append(user_func(x_star))
		delta_x = -inv_B.dot(analytical_grad(x_star))
		t = backtrackline(x_star,delta_x)
		x_star_1 = np.add(x_star,t*delta_x)
		y = analytical_grad(x_star_1) - analytical_grad(x_star)
		s = x_star_1 - x_star
		numerator = np.dot(y,np.transpose(y))
		numerator = np.dot(inv_B,numerator)
		numerator = np.dot(numerator,inv_B)
		inv_B = inv_B + 1/np.dot(np.transpose(s),y)*np.dot(s,np.transpose(s)) - (1/np.dot((np.dot(np.transpose(y),inv_B)),y))*numerator
		if user_func(x_star) - user_func(x_star_1) < tol:
			x_star_1 = x_star
			break
		x_star = x_star_1

	return x_star,x1,x2,fx,iterations	

def analytical_grad(x_star):

	y = np.zeros([x_star.shape[0],1])
	y[0] = 200*(x_star[1]-x_star[0]**2)*2*-1*x_star[0] - 2*(1-x_star[0])
	y[1] = 200*(x_star[1]-x_star[0]**2)
	return y

x_star = np.array([[1.5],[1.5]])
N = 15000
tol = 1e-8

x_star,x1,x2,fx,iterations = BFGS(N,tol,x_star)
print x_star
import matplotlib.pyplot as plt 

plt.plot(iterations,x1)
plt.plot(iterations,x2)
plt.xlabel("iterations")
plt.ylabel("x value")
plt.show()
plt.plot(iterations,fx)
plt.ylabel("function value")
plt.xlabel("iterations")
plt.show()

x_star = np.array([[1.5],[1.5]])
N = 15000
tol = 1e-8

x_star,x1,x2,fx,iterations = DFP(N,tol,x_star)

print x_star
import matplotlib.pyplot as plt 

plt.plot(iterations,x1)
plt.plot(iterations,x2)
plt.xlabel("iterations")
plt.ylabel("x value")
plt.show()
plt.plot(iterations,fx)
plt.ylabel("function value")
plt.xlabel("iterations")
plt.show()
