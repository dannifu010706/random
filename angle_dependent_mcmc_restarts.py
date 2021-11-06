#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:06:11 2021


"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:06:08 2020

"""

# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
import os
#==============================================================================

# OBJECTIVE FUNCTION
    
def subroutine_scat(x,theta,lam):  
    # theta is one of the coordinate angles [-pi/2,pi/2]
    # lam is the wavelength (0.4-0.7 microns)
    
    # SYSTEM PARAMETERS
    
    n = x.shape[0]
    
    rho = np.ones([n])
    
    omega = 1.0
    
    #incoming wave angles
    phi = 0.0 * np.pi # can range from [0, 2pi)
    
    # susceptibility
    chiii = 10.0 + 1j
    
    # frequency
    k = (2.0 * np.pi)/lam # 1/microns
    
    # volume
    V_0 = (0.05)**3 # microns^3
    
    # incoming wave vector
    K = k * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)], [np.cos(theta)]],complex)
    
    # polarization vector
    vecinc = np.array([[np.cos(theta)], [np.cos(theta)], [-(np.sin(theta) * (np.cos(phi) + np.sin(phi)))]], complex) 
    # (any vector perpendicular to K)
    
    # for the fixed alpha case, Clausius-Mossoti factor parameter value(s)
    alpha = (V_0 * 3.0 * chiii)/(chiii + 3.0)
    
    # the 3 x 3 matrix that determines an isotropy of the scatterer orientation
    A_matrix = np.identity(3) # could be any symmetric matrix, which determines the (an)isotropy of the scatterer orientation

    def a_j(r, a, A): # the Claussius-Mossotti factor, determined by a symmetric (3 × 3) matrix such that (A_i)^T = A_i
        alph = np.array([[0,0,0],[0,0,0],[0,0,0]],complex)
        for i in range(3):
            for j in range(3):
                alph[i,j] = (r * a * A[i,j])
        return alph
 
    def E_inc_j(tilde_k, x, vecinc): # the incoming electric field
        n = x.shape[0]
        e = np.zeros([3*n,1],complex)
        for i in range(n):
            block_result = np.exp(1j * np.vdot(x[i], tilde_k)) * vecinc
            for j in range(3):
                e[3*i + j, 0] = block_result[j,0]
        return e # returns a matrix containing the incoming electric field specific to each scatterer, i
                 
    def G(u, k): # the the (3 × 3) Green’s function for Maxwell on R^3 for a given frequency k
        a = 1/((np.abs(k))**2)
        
        u1 = u[0]
        u2 = u[1]
        u3 = u[2]
        g = np.array([[0,0,0],[0,0,0],[0,0,0]],complex)
        
        g[0,0] = 1 + (a * ((-(k**2)*(u1**2)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u1**2)*(np.linalg.norm(u)**(-3))) + (1j*k*(np.linalg.norm(u)**(-1)))
                 - (np.linalg.norm(u)**(-2)) + (3*(u1**2)*(np.linalg.norm(u)**(-4)))))
        
        g[0,1] = a * ((-(k**2)*(u1*u2)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u1*u2)*(np.linalg.norm(u)**(-2))) + (3*(u1*u2)*(np.linalg.norm(u)**(-4))))
        
        g[0,2] = a * ((-(k**2)*(u1*u3)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u1*u3)*(np.linalg.norm(u)**(-2))) + (3*(u1*u3)*(np.linalg.norm(u)**(-4))))
        
        g[1,0] = a * ((-(k**2)*(u2*u1)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u2*u1)*(np.linalg.norm(u)**(-2))) + (3*(u2*u1)*(np.linalg.norm(u)**(-4))))
        
        g[1,1] = 1 + (a * ((-(k**2)*(u2**2)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u2**2)*(np.linalg.norm(u)**(-3))) + (1j*k*(np.linalg.norm(u)**(-1)))
                 - (np.linalg.norm(u)**(-2)) + (3*(u2**2)*(np.linalg.norm(u)**(-4)))))
        
        g[1,2] = a * ((-(k**2)*(u2*u3)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u2*u3)*(np.linalg.norm(u)**(-2))) + (3*(u2*u3)*(np.linalg.norm(u)**(-4))))
        
        g[2,0] = a * ((-(k**2)*(u3*u1)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u3*u1)*(np.linalg.norm(u)**(-2))) + (3*(u3*u1)*(np.linalg.norm(u)**(-4))))
        
        g[2,1] = a * ((-(k**2)*(u3*u2)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u3*u2)*(np.linalg.norm(u)**(-2))) + (3*(u3*u2)*(np.linalg.norm(u)**(-4))))
        
        g[2,2] = 1 + (a * ((-(k**2)*(u3**2)*(np.linalg.norm(u)**(-2))) - 
                 (3*1j*k*(u3**2)*(np.linalg.norm(u)**(-3))) + (1j*k*(np.linalg.norm(u)**(-1)))
                 - (np.linalg.norm(u)**(-2)) + (3*(u3**2)*(np.linalg.norm(u)**(-4)))))
        
        g = np.exp(1j * k * abs(u))
    
        return g
    
    #EXTINCTION
    
    def W_ext(x, k, rho, alpha, A): # particle–particle interaction term
        n = x.shape[0] # the number of x vextors 
        result = np.zeros([3*n,3*n],complex)
        u = np.zeros((n, 3)) # u = x - x' 
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    u[i] = x[i] - x[j]
                    block_result = a_j(rho[i], alpha, A) * G((u[i]), k) * a_j(rho[j], alpha, A) 
                    for m in range(3):
                        for l in range(3):
                            result[3*i + m, 3*j + l] = block_result[m,l]
        return result.imag
    
    def A_ext(rho, a, A): # single-particle term
        n = rho.shape[0]
        result = np.zeros([3*n,3*n],complex)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    block_result = a_j(rho[i], a, A).imag
                    for m in range(3):
                        for l in range(3):
                            result[3*i + m, 3*j + l] = block_result[m,l]
        return result # (3 x 3) matrix
    
    def P_ext(e, A, W, omega):
        eT = np.matrix.getH(e)
        mm1 = np.matmul(A, e)
        mm2 = np.matmul(W, e)
        extinction = (np.dot(eT, mm1) + np.dot(eT, mm2)) * (omega/2.0)
        return extinction
    
    #ABSORPTION
    
    def W_abs(x, k, rho, alpha, A, chi): # particle–particle interaction term
        n = x.shape[0] 
        result = np.zeros([3*n,3*n],complex)
        u = np.zeros((n, 3)) 
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    u[i] = x[i] - x[j]
                    block_result = np.matrix.getH(a_j(rho[i], alpha, A)) * (1.0 / np.conjugate(chi)).imag * a_j(rho[i], alpha, A) * G((u[i]), k) * a_j(rho[j], alpha, A) 
                    for m in range(3):
                        for l in range(3):
                            result[3*i + m, 3*j + l] = block_result[m,l]                   
        return 2.0 * result.real  # (3 x 3) matrix

    def A_abs(rho, a, A, chi): # single-particle term
        n = rho.shape[0]
        result = np.zeros([3*n,3*n],complex)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    block_result = np.matrix.getH(a_j(rho[i], a, A)) * (1.0 / np.conjugate(chi)).imag * a_j(rho[i], a, A)
                    for m in range(3):
                        for l in range(3):
                            result[3*i + m, 3*j + l] = block_result[m,l]
        return result # (3 x 3) matrix

    def P_abs(e, A, W, omega):
        eT = np.matrix.getH(e)
        mm1 = np.matmul(A, e)
        mm2 = np.matmul(W, e)
        absorption = (np.dot(eT, mm1) + np.dot(eT, mm2)) * (omega/2.0)
        return absorption

    # SCATTERED
    
    def P_scat(P_ext, P_abs):
        return P_ext - P_abs
    
    # RETURNS

    extinction = P_ext(E_inc_j(K, x, vecinc), A_ext(rho, alpha, A_matrix),
                       W_ext(x, k, rho, alpha, A_matrix), omega)

    absorption = P_abs(E_inc_j(K, x, vecinc), A_abs(rho, alpha, A_matrix, chiii), 
                       W_abs(x, k, rho, alpha, A_matrix, chiii), omega)
    
    scattered = P_scat(extinction, absorption)
    
    return float(scattered.real)

def Gaussian(x,alpha,r):
    return 1./(np.sqrt(alpha**np.pi))*np.exp(-alpha*np.power((x - r), 2.))

def FOM(x): 
    theta_i = -np.pi/2 + 0.01 * np.pi #lower theta bound
    theta_f = np.pi/2 - 0.01 * np.pi #upper theta bound
    lam_i = 0.5 #lower lambda bound
    lam_f = 0.565 #upper lambda bound
    delta_theta = 0.1 #step-size for theta in Riemann Sum
    delta_lam = 0.1 #step-size for lambda in Riemann Sum
    lam_naught = 0.5 #wavelength that target Gaussian is center
    theta_step = np.arange(theta_i,theta_f,delta_theta) #creating array of theta values for Riemann Sum
    lam_step = np.arange(lam_i,lam_f,delta_lam) #creating array of lambda values for Riemann Sum
    P_naught = 0.0 #Initializing P_naught, a normalization factor
    figure_of_merit = 0.0 #initializing Riemann Sum
    P_scat = []
    P_target = []
    for i in range(len(lam_step)):
        for j in range(len(theta_step)):
            P_scat.append(subroutine_scat(x,((theta_step[j-1]+theta_step[j])/2),((lam_step[i-1]+lam_step[i])/2))) 
            P_target.append(Gaussian((lam_step[i-1]+lam_step[i])/2,1,lam_naught))
    P_naught = max(P_scat) * ((np.pi) ** (1/2))
    P_scat = np.asarray(P_scat)
    P_target = np.asarray(P_target)
    P_scat = P_scat / P_naught
    integrand = np.subtract(P_scat,P_target)
    integrand = np.abs(integrand) ** 2
    figure_of_merit = np.sum(integrand) * delta_lam * delta_theta
    return figure_of_merit
#==============================================================================
    
# MCMC(fTAR, delta, N, T):
# Returns X

# input : fTAR   : objective function
#         delta  : maximum random stepsize for path modification
#         N      : number of time steps
#         T      : maximum number of path modifications
# output: X      : a (N, 3) position vector array of the optimized scatterer configuration
    
#------------------------------------------------------------------------------
    
def MCMC(fTAR, delta, N, T):
    start = time.time()
    xxx = []
    
    path_to_files = "/Users/alexmcentarffer/Data/X_angle_dependent/X_data51"

    def x(file_path):
        with open(str(file_path), "r") as file:
            lines = file.readlines()
            lines = np.asarray(lines)
            x = np.zeros((25,3))
            for i in range(25):
                x[i,:] = [float(x) for x in lines[i].split(' ')]
        return x

    #print(x(N))
    x_0 = x(path_to_files)
    def x_plot(x):
        fig = plt.figure()
        fig.add_subplot(111)
        X = x[0:,0]
        Y = x[0:,1]
        array = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xticks(array, array, rotation = 'horizontal')
        plt.yticks(array, array, rotation = 'horizontal')
        plt.plot(X, Y, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)        
        plt.show()
        
    #print(x_plot(x_0))
    
    definitely_accept = 0
    accept_w_prob = 0
    reject = 0
    for i in range(T):
       
        # ITERATION 
        
        x_1 = np.zeros([N, 3])
        for i in range(N):
            x_1[i,0] = x_0[i,0] + (delta * np.random.uniform(-1.0, 1.0)) # random samples 
            # are drawn from a uniform distribution over the half-open interval: [-1, 1)
            x_1[i,1] = x_0[i,1] + (delta * np.random.uniform(-1.0, 1.0)) 
        
        for i in range(N): # make sure x is contrained in [1, 1, 1] box
            for j in range(2):
                if x_1[i, j] < 0.0:
                    x_1[i, j] = 0.0    
                if x_1[i, j] > 1.0:
                    x_1[i, j] = 1.0
                
        if (np.abs(fTAR(x_1) / fTAR(x_0)) < 1): # definitely accept
            X = x_1 
            x_0 = x_1
            xxx.append(X)
            definitely_accept = definitely_accept + 1
        else:
            u = np.random.rand(1) 
            # beta = 3500000 --> 0.311553
            alpha = np.exp(-0.5 * 3500000. * (np.abs(fTAR(x_1)) - np.abs(fTAR(x_0))))
            # print(u)
            # print(alpha)
            if u <= alpha: # accept with ~30% probability
                # print('True')
                X = x_1 
                x_0 = x_1
                xxx.append(X)
                accept_w_prob = accept_w_prob + 1
            else:
                reject = reject + 1
    print(x_plot(X))
    print(X)
    values = 'X:' + str(X)
    values += '\n'
    values += 'number of times we definitely accept:' + str(definitely_accept)
    values += '\n'
    values += 'number of times we happen to accept w/ probability:' + str(accept_w_prob)
    values += '\n'
    values += 'number of times we reject:' + str(reject)
    values += '\n'
    values += 'fTAR(X):' + str(fTAR(X))
    
    t = definitely_accept + accept_w_prob
    y = np.zeros([t])
    i = 0
    for i in range(t):
        func_value = float(fTAR(xxx[i]))
        y[i] = (func_value)
    #print(y)
    # Distribution = 'RHO Distribution:' + str(xxx)
   # Distribution = 'fTAR Distribution:' + str(y)
    #print(Distribution)
    x = np.arange(0, t, 1)
    plt.plot(x, y, '-')
    plt.title('Energy vs. Time')
    plt.xlabel('MCMC step')
    plt.ylabel('fTAR')
    plt.grid(True)
    plt.show()
    end = time.time()
    print(end - start)
    #print(values)

    return X

# =============================================================================

# CALLING
 
fTAR = FOM
delta = 0.01 #increase
N = 25
T = 1000
X = MCMC(fTAR, delta, N, T)
