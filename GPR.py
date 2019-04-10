import numpy as np
import os
np.random.seed(123)

PATH = 'E:/AnacondaProjects/GPR'
os.chdir(PATH)

# X is training data, Y is target data        
def load_data(X_csv, X_test_csv, Y_csv):
    X = np.loadtxt(X_csv, delimiter=',', skiprows=0, encoding='utf-8_sig')
    X = np.array(X)

    X_test = np.loadtxt(X_test_csv, delimiter=',', skiprows=0, encoding='utf-8_sig')
    X_test = np.array([X_test])

    Y = np.loadtxt(Y_csv, delimiter=',', skiprows=0, encoding='utf-8_sig')
    Y = np.array([Y]).T
    return X, X_test, Y

X, X_test, Y = load_data('X.csv', 'X_test.csv', 'Y.csv')

# zscore normalization
def zscore(X, X_test, Y):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
    
    X_test_norm = (X_test - X_mean) / X_std 
            
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    Y_norm = (Y - Y_mean) / Y_std
    
    X_norm_m = np.mat(X_norm)
    X_test_norm_m = np.mat(X_test_norm)
    Y_norm_m = np.mat(Y_norm)
    return X_norm_m, X_test_norm_m, Y_norm_m

X_norm_m, X_test_norm_m, Y_norm_m = zscore(X, X_test, Y)


###### Gaussian Process ######        
def kernel(x, x_dash, theta):
    k_xx = theta[0] * np.exp( - np.sqrt(np.sum(np.power(x - x_dash, 2)))**2 / theta[1]) #RBF
    return k_xx

def kernel_matrix(theta):        
    K_nonnoise = np.zeros(shape=(X_norm_m.shape[0], X_norm_m.shape[0]))
    for i in range(X_norm_m.shape[0]):
        for j in range(X_norm_m.shape[0]):
            k = kernel(X_norm_m[i,:], X_norm_m[j,:], theta)
            K_nonnoise[i,j] = k
        
    noise = np.mat(np.eye(N=K_nonnoise.shape[0], M=K_nonnoise.shape[1]) * theta[2])
    K =  K_nonnoise + noise
    K_m = np.mat(K)
    return K_m
   
# log-likelihood
def log_likelihood(theta):
    K_m = kernel_matrix(theta)
    L = - np.log(np.linalg.det(K_m)) - Y_norm_m.T * np.linalg.inv(K_m) * Y_norm_m
    return L

# graddient log-likelihood
def grad_log_likelihood(theta):
    K_m = kernel_matrix(theta)
    delta = np.mat(np.eye(N=K_m.shape[0], M=K_m.shape[1]))
    
    tau = np.log(theta[0])
    sigma = np.log(theta[1])
    eta = np.log(theta[2])
    
    pow_x_x_dash = np.zeros(shape=(X_norm_m.shape[0], X_norm_m.shape[0]))
    for i in range(X_norm_m.shape[0]):
        for j in range(X_norm_m.shape[0]):
            distance = np.sqrt(np.sum(np.power(X_norm_m[i,:] - X_norm_m[j,:], 2)))
            pow_x_x_dash[i,j] = distance ** 2

    grad_K_tau = K_m - np.exp(eta) * delta
    grad_K_sigma = (K_m - np.exp(eta) * delta) * np.exp(-sigma) * pow_x_x_dash
    grad_K_eta = np.exp(eta) * delta

    grad_L_tau = - np.trace(np.linalg.inv(K_m) * grad_K_tau) + (np.linalg.inv(K_m) * Y_norm_m).T * grad_K_tau * (np.linalg.inv(K_m) * Y_norm_m)
    grad_L_sigma = - np.trace(np.linalg.inv(K_m) * grad_K_sigma) + (np.linalg.inv(K_m) * Y_norm_m).T * grad_K_sigma * (np.linalg.inv(K_m) * Y_norm_m)
    grad_L_eta = - np.trace(np.linalg.inv(K_m) * grad_K_eta) + (np.linalg.inv(K_m) * Y_norm_m).T * grad_K_eta * (np.linalg.inv(K_m) * Y_norm_m)

    grad_L_tau = np.array(grad_L_tau).ravel()
    grad_L_sigma = np.array(grad_L_sigma).ravel()
    grad_L_eta = np.array(grad_L_eta).ravel()
    
    grad_L = np.array([grad_L_tau, grad_L_sigma, grad_L_eta])
    return grad_L


#%%

import pickle
from tqdm import tqdm

upper = 50
tick = 0.1
	
theta_1 = np.array([np.arange(0.1, upper+tick, tick)])
theta_2 = np.array([np.arange(0.1, upper+tick, tick)])
theta_3 = np.array([np.arange(0.1, upper+tick, tick)])

'''
n = int(upper / tick)
Z = np.zeros(shape=(n,n,n))
for i in tqdm(range(n)):
    for ii in range(n):
        for iii in range(n):
            theta = np.array([theta_1[0,i], theta_2[0,ii], theta_3[0,iii]])
            Z[i,ii,iii] = log_likelihood(theta)

f = open('Z.pickle','wb')
pickle.dump(Z,f)
f.close
'''
#%%
import pickle

f = open('E:/AnacondaProjects/heavy_pickle/GPR/Z.pickle','rb')
Z = pickle.load(f)

Z_max_pos = np.unravel_index(np.argmax(Z), Z.shape)
Z_max = Z[Z_max_pos[0], Z_max_pos[1]]

#%%
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, Y, Z = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]

T = np.exp(-X**2 - Y**2 - Z**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(X, Y, Z, c=Z.flatten(), alpha=0.5)
fig.colorbar(scat, shrink=0.5, aspect=5)


X, Y, Z = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]

T = np.exp(-X**2 - Y**2 - Z**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(X, Y, Z, c=Z.flatten(), alpha=0.5)
fig.colorbar(scat, shrink=0.5, aspect=5)
#%%
import matplotlib.pyplot as plt
import pickle

f = open('Z.pickle','rb')
Z = pickle.load(f)

fig, ax = plt.subplots(1,1, figsize=(16,9))
ax.imshow(Z,interpolation='nearest',vmin=-30,vmax=1,cmap='inferno')
ax.grid(True)
plt.colorbar()
plt.savefig('heatmap.png', dpi=300)
plt.show()

#%%
'''
# gradient descent
def gradient_descent(theta_init, learning_rate, iteration_max):
    theta = theta_init
    for i in range(iteration_max):
        grad_L = grad_log_likelihood(theta).reshape(3)
        theta_new = theta - learning_rate * grad_L
    return theta_new

theta_init = np.random.rand(3)
learning_rate = 0.001
iteration_max = 100
theta_opt = gradient_descent(theta_init, learning_rate, iteration_max)

print('log-likelihood: {}'.format(log_likelihood(theta_opt)))
'''
#%%
# expected predictive distribution
theta = np.array([0.1*Z_max_pos[0], 0.1*Z_max_pos[1], 0])
C_nonnoise = np.zeros(shape=(X_norm_m.shape[0]+X_test_norm_m.shape[0], X_norm_m.shape[0]+X_test_norm_m.shape[0]))
stack_X = np.concatenate([X_norm_m, X_test_norm_m], axis=0)
for i in range(stack_X.shape[0]):
    for j in range(stack_X.shape[0]):
        k = kernel(stack_X[i,:], stack_X[j,:], theta)
        C_nonnoise[i,j] = k
        
noise = np.mat(np.eye(N=C_nonnoise.shape[0], M=C_nonnoise.shape[1]) * theta[2])
C =  C_nonnoise + noise

K = C[:X_norm_m.shape[0], :X_norm_m.shape[0]]
k_ast = C[:X_norm_m.shape[0], X_norm_m.shape[0]:]
k_ast_ast = C[X_norm_m.shape[0]+X_test_norm_m.shape[0]-1, X_norm_m.shape[0]+X_test_norm_m.shape[0]-1]
K_m = np.mat(K)
k_ast_m = np.mat(k_ast)

E_y = k_ast_m.T * np.linalg.inv(K_m) * Y_norm_m

Y_mean = np.mean(Y, axis=0)
Y_std = np.std(Y, axis=0)
    
Y_pred = E_y * Y_std + Y_mean

print(Y_pred)