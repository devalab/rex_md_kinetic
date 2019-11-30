import numpy as np
from scipy.stats import multivariate_normal
import auxillary_data_structures as aux
import numpy as np
import time,math,sys
from scipy import linalg

def compute_precisions_chol(covariances, reg_covar=1e-6):
	precisions_chol = []
	log_determinants = []
	for matrix in covariances:
		size = len(matrix)
		try:
			cov_chol = linalg.cholesky(matrix, lower=True)
		except linalg.LinAlgError:
			#regularizing the covariance matrix
	    		matrix += reg_covar*np.eye(matrix.shape[0])
			cov_chol = linalg.cholesky(matrix, lower=True)
		precisions_chol.append(linalg.solve_triangular(cov_chol, np.eye(size),lower=True).T)
		log_determinants.append(np.sum(np.log(precisions_chol[-1].diagonal())))
	return np.array(precisions_chol), np.array(log_determinants)

def log_pdf_multivariate(x, mu, sigma, precisions_chol, log_determinant):
    size = x.shape[1]
    
    if size == mu.shape[0] and (size, size) == sigma.shape:
	
	log_det = log_determinant

        norm_const = float(size) * np.log(2 * np.pi)

        x_mu = np.matrix(x - mu)
        
	exp_power = np.dot(x_mu,precisions_chol)
	log_exp = []
	for i in xrange(x.shape[0]):
		log_exp.append(float(np.dot(exp_power[i,:],exp_power[i,:].T)))
	log_exp = np.array(log_exp)
	
        result = log_exp
	
        return ((norm_const + result) *- 0.5) + (log_det*1.0)
    else:
        raise NameError("The dimensions of the input don't match")

def log_pdf(data, mean, covariance, beta, energy_data, energy_cluster, prec_chol, log_determinant):
	#num = -beta*(energy_data)
	#max_val = max(-beta*(energy_data),-beta*(energy_cluster))
	#denom = max_val + np.log(np.exp(-beta*(energy_data) - max_val) + np.exp(-beta*(energy_cluster) - max_val))
	prob = log_pdf_multivariate(x=data, mu=mean, sigma=covariance, precisions_chol=prec_chol, log_determinant=log_determinant)

	for i in xrange(energy_data.shape[0]):
		if energy_data[i] > energy_cluster:
			prob[i] += np.log(beta) - beta*(energy_data[i] - energy_cluster)
	
	return prob
