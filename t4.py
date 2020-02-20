#Name: Vipul Ramtekkar 
#Rollno: 16D110013
#Assignment 4

import numpy as np 
import math 
import matplotlib.pyplot as plt 


def plot_figures(x_final,x1,x2,fx,iterations,A):

	print x_final
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

def steepest_decent(N,tol,x_star):
	x1 = []
	x2 = []
	iterations = []
	fx = []

	for i in range(N):
		x1.append(x_star[0])
		x2.append(x_star[1])
		iterations.append(i)
		fx.append(user_func(x_star))
		delta_x = -analytical_grad(x_star)
		alpha = backtrackline(x_star,delta_x)
		x_star_1 = x_star + alpha*delta_x

		if user_func(x_star) - user_func(x_star_1) < tol:
			x_star_1 = x_star
			break
		x_star = x_star_1

	return x_star,x1,x2,fx,iterations

def FRCG(N,tol,x_star):
	x1 = []
	x2 = []
	iterations = []
	fx = []

	for i in range(N):
		x1.append(x_star[0])
		x2.append(x_star[1])
		iterations.append(i)
		fx.append(user_func(x_star))
		delta_x = -analytical_grad(x_star)
		if i == 0:
			beta = 1
			delta_x_old = delta_x
		else:
			beta = np.dot(np.transpose(delta_x),delta_x)/np.dot(np.transpose(delta_x_old),delta_x_old)
		if i == 0:
			s = delta_x
		else:
			s = delta_x + beta*s_old
		
		alpha = backtrackline(x_star,s)
		x_star_1 = x_star + alpha*s
		if user_func(x_star) - user_func(x_star_1) < tol:
			x_star_1 = x_star
			break
		x_star = x_star_1
		s_old = s
		delta_x_old = delta_x

	return x_star,x1,x2,fx,iterations

def analytical_grad(x_star):

	y = np.zeros([x_star.shape[0],1])
	y[0] = 200*(x_star[1]-x_star[0]**2)*2*-1*x_star[0] - 2*(1-x_star[0])
	y[1] = 200*(x_star[1]-x_star[0]**2)
	return y

x_star = np.array([[1.5],[1.5]])
N = 15000
tol = 1e-8
x_final_BFGS,x1_BFGS,x2_BFGS,fx_BFGS,iterations_BFGS = BFGS(N,tol,x_star)
x_final_DFP,x1_DFP,x2_DFP,fx_DFP,iterations_DFP = DFP(N,tol,x_star)
x_final_SD,x1_SD,x2_SD,fx_SD,iterations_SD = steepest_decent(N,tol,x_star)
plot_figures(x_final_SD,x1_SD,x2_SD,fx_SD,iterations_SD,"Steepest Descent")
x_final_FRCG,x1_FRCG,x2_FRCG,fx_FRCG,iterations_FRCG = FRCG(N,tol,x_star)
plot_figures(x_final_FRCG,x1_FRCG,x2_FRCG,fx_FRCG,iterations_FRCG, "FRCG")
x1 = [x1_BFGS,x1_DFP,x1_SD,x1_FRCG]
x2 = [x2_BFGS,x2_DFP,x2_SD,x2_FRCG]
i = [iterations_BFGS,iterations_DFP,iterations_SD,iterations_FRCG]
fx = [fx_BFGS,fx_DFP,fx_SD,fx_FRCG]

fig, axs = plt.subplots(2,2)
axs[0,0].plot(i[0],x1[0],i[0],x2[0])
axs[0,0].set_title("BFGS")
axs[0,0].set_ylabel("X")
axs[0,0].legend(("x1","x2"))
axs[0,1].plot(i[1],x1[1],i[1],x2[1])
axs[0,1].set_title("DFP")
axs[0,1].legend(("x1","x2"))
axs[1,0].plot(i[2],x1[2],i[2],x2[2])
axs[1,0].set_title("Steepest Descent")
axs[1,0].set_ylabel("X")
axs[1,0].legend(("x1","x2"))
axs[1,0].set_xlabel("Iterations")
axs[1,1].plot(i[3],x1[3],i[3],x2[3])
axs[1,1].set_title("FRCG")
axs[1,1].set_xlabel("Iterations")
axs[1,1].legend(("x1","x2"))

plt.tight_layout()
plt.show()
plt.close()


fig, axs = plt.subplots(2,2)
axs[0,0].plot(i[0],fx[0])
axs[0,0].set_title("BFGS")
axs[0,0].set_ylabel("F(X)")
axs[0,1].plot(i[1],fx[1])
axs[0,1].set_title("DFP")
axs[1,0].plot(i[2],fx[2])
axs[1,0].set_title("Steepest Descent")
axs[1,0].set_ylabel("F(X)")
axs[1,0].set_xlabel("Iterations")
axs[1,1].plot(i[3],fx[3])
axs[1,1].set_title("FRCG")
axs[1,1].set_xlabel("Iterations")

plt.tight_layout()
plt.show()
plt.close()
