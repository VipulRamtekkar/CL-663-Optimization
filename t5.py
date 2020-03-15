#Name: Vipul Ramtekkar 
#Rollno: 16D110013
#Assignment 5
#Compatible with both python 2 and python 3

import numpy as np 
import math 
import matplotlib.pyplot as plt 


def plot_figures(x_final,x1,x2,fx,iterations,A):

	print (x_final)
	plt.title(A)
	plt.plot(iterations,x1)
	plt.plot(iterations,x2)
	plt.legend(("x1","x2"))
	plt.xlabel("Iterations")
	plt.ylabel("X")
	plt.show()
	plt.title(A)
	plt.plot(iterations,fx)
	plt.ylabel("F(X)")
	plt.xlabel("Iterations")
	plt.show()

def user_func(x_star):
	y = 100*(x_star[1] - x_star[0]**2 )**2 + (1 - x_star[0])**2
	return y 

def hessian(x_star):
	y = np.zeros((2,2))
	y[0,0] = -400*x_star[1] + 1200*(x_star[0]**2) + 2
	y[0,1] = -400*x_star[0]
	y[1,0] = -400*x_star[0]
	y[1,1] = 200
	return y

def analytical_grad(x_star):

	y = np.zeros([x_star.shape[0],1])
	y[0] = 200*(x_star[1]-x_star[0]**2)*2*-1*x_star[0] - 2*(1-x_star[0])
	y[1] = 200*(x_star[1]-x_star[0]**2)
	return y

def approx_function(x_star,pk):
	a = user_func(x_star) + np.dot(np.transpose(analytical_grad(x_star)),pk) + 0.5 * np.dot(np.dot(np.transpose(pk),hessian(x_star)),pk)
	return a[0]
def TrustRegion(x_star, delta_knot, delta_norm, N, tol, eta):
	# Cauchy Point Step 
	x1 = []
	x2 = []
	iterations = []
	fx = []
	for i in range(N):
		x1.append(x_star[0])
		x2.append(x_star[1])
		iterations.append(i)
		fx.append(user_func(x_star))
		cauchy_criteria = np.dot(np.dot(np.transpose(analytical_grad(x_star)), hessian(x_star)),analytical_grad(x_star))

		if cauchy_criteria > 0:
			p_c = - (np.dot(np.transpose(analytical_grad(x_star)),analytical_grad(x_star))/cauchy_criteria)*analytical_grad(x_star)
		else:
			p_c = -(delta_knot/np.linalg.norm(analytical_grad(x_star)))*analytical_grad(x_star)
		p_n = -np.dot(np.linalg.inv(hessian(x_star)),analytical_grad(x_star))

		if np.linalg.norm(p_n) < delta_knot or np.linalg.norm(p_n) == delta_knot:
			pk = p_n
		elif np.linalg.norm(p_c) > delta_knot or np.linalg.norm(p_c) == delta_knot:
			pk = (delta_knot/np.linalg.norm(p_c))*p_c
		else:
			pk = eta*p_n + (1-eta)*p_c

		x_del = x_star + pk
		zero = np.array([[0],[0]])

		rho_k = (user_func(x_star) - user_func(x_del))/(approx_function(x_star,zero) - approx_function(x_star,pk))

		if rho_k < 0.25:
			delta_knot = 0.25*np.linalg.norm(pk)
		else:
			if rho_k > 0.75 and np.linalg.norm(pk) == delta_knot:
				delta_knot = min(2*delta_knot, delta_norm)
		if rho_k > eta:
			x_star_1 = x_star + pk
		else:
			x_star_1 = x_star
			continue

		if user_func(x_star) - user_func(x_star_1) < tol:
			x_star_1 = x_star
			break
		x_star = x_star_1

	return x_star,x1,x2,fx,iterations

x_star = np.array([[1.5],[1.5]])
N = 15000
tol = 1e-8
eta = 0.2 
delta_knot = 0.5
delta_norm = 1
x_final,x1,x2,fx,iterations = TrustRegion(x_star, delta_knot, delta_norm, N, tol,eta)
plot_figures(x_final,x1,x2,fx,iterations,"Trust Region Approach")
