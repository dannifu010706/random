#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:34:24 2020


"""

import numpy as np
from scipy.optimize import minimize, check_grad

# GLOBAL VARIABLE: SCATTERER POSITIONS

def x(n): # this initialises the dense square lattice grid of scatterer locations
# returns (n x n) scatterers
    m = int(np.sqrt(n)) # the way this is set up: n MUST BE A PERFECT SQUARE
    nx, ny = (m, m)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    
    x = np.zeros((m**2, 3))
    flag = 0
    for i in range(m):
        for j in range(m):
            x[flag,:] = np.array([xv[i,j], yv[i,j], 0])
            flag = flag + 1
    return x # a (n x 3) matrix of describing the locations of each scatterer's position, (x, y, 0)

# OBJECTIVE FUNCTION

def subroutine_scat(rho,theta,lam):  
    #lam is the wavelength (values between 0.4-0.7 microns)
    #theta one of the spherical angles [-pi/2,pi/2]
    # SYSTEM PARAMETERS
    
    omega = 1.0
    
    # incoming wave angles
    phi = 0.0 # can range from [0, 2pi)
    
    # susceptibility
    chiii = 10.0 + 1j
    
    # frequency
    k = (2.0 * np.pi)/lam # 1/microns
    
    # volume
    V_0 = (0.05)**3 # microns^3
    
    # incoming wave vector
    K = k * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)], [np.cos(theta)]],complex)
    
    # polarization vector
    vecinc = np.array([[1], [1], [-(np.tan(theta) * (np.cos(phi) + np.sin(phi)))]], complex) 
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
        n = x.shape[0] # the number of x vectors 
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
        return result.imag # (3 x 3) matrix
    
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
        return 2.0 * result.real # (3 x 3) matrix

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
    
    n = rho.shape[0]
    
    extinction = P_ext(E_inc_j(K, x(n), vecinc), A_ext(rho, alpha, A_matrix),
                       W_ext(x(n), k, rho, alpha, A_matrix), omega)

    absorption = P_abs(E_inc_j(K, x(n), vecinc), A_abs(rho, alpha, A_matrix, chiii), 
                       W_abs(x(n), k, rho, alpha, A_matrix, chiii), omega)
    
    scattered = P_scat(extinction, absorption)
    
    return float(scattered.real)

def Gaussian(x,alpha,r):
    return 1./(np.sqrt(alpha**np.pi))*np.exp(-alpha*np.power((x - r), 2.))

def FOM(rho): 
    theta_i = -np.pi/4 #lower theta bound
    theta_f = np.pi/4 #upper theta bound
    lam_i = 0.4 #lower lambda bound
    lam_f = 0.7 #upper lambda bound
    delta_theta = 0.1 #step-size for theta in Riemann Sum
    delta_lam = 0.1 #step-size for lambda in Riemann Sum
    lam_naught = 0.5 #wavelength that target Gaussian is center
    theta_step = np.arange(theta_i,theta_f,delta_theta) #creating array of theta values for Riemann Sum
    lam_step = np.arange(lam_i,lam_f,delta_lam) #creating array of lambda values for Riemann Sum
    P_naught = 0.0 #Initializing P_naught, a normalization factor
    for i in range(len(lam_step)):
        for j in range(len(theta_step)):
            if P_naught < subroutine_scat(rho,theta_step[j],lam_step[i]):
                P_naught = subroutine_scat(rho,theta_step[j],lam_step[i]) #want P_naught to be maximum value of scattered power
    print(P_naught)
    figure_of_merit = 0.0 #initializing Riemann Sum
    for i in range(len(lam_step)):
        for j in range(len(theta_step)):
            P_scat = subroutine_scat(rho,theta_step[j],lam_step[i]) / P_naught
            P_target = Gaussian(lam_step[i],1,lam_naught)
            integrand = P_scat - P_target
            figure_of_merit = figure_of_merit + (np.abs(integrand)) **2
    return figure_of_merit
            

# TESTS
    
def rho(n):
    return np.random.rand(n, 1)

rho0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.8, 0.9])

# TEST = gradient_scat(rho0)
TEST2 = FOM(rho0)
print(TEST2)
# print(TEST)

# -----------------------------------------------------------------------------

# MINIMIZATION    

#bnds = []
#for i in rho0:
    #bnds.append((0.0, 1.0))
#bnds = np.array(bnds) # constrains rho to values between 0 and 1 only

#opt = minimize(subroutine_scat, rho0, method='SLSQP', bounds=bnds, tol=1e-12, options={'maxiter': 10000000000, 'iprint': 1, 'disp': False})
#print(opt)

# -----------------------------------------------------------------------------

# CHECKING THE GRADIENT

# gradientChecker = check_grad(subroutine_scat, gradient_scat, rho0)
# print(gradientChecker)
