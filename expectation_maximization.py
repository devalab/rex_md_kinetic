from scipy.stats import multivariate_normal
import auxillary_data_structures as aux
import numpy as np
import time,math,sys
from scipy import linalg
from scipy import optimize
import shelve
import prob_distributions as prob_dis	

class Expectation_Maximization():
    #param_grid is a dictionary having lists of all the respective prameters for the probability-distribution
    def __init__(self, param_grid, reg_covar=1e-6, alpha=0.05, threshold = 1e-3, iterations=50):
        self.data = np.array(param_grid["data"])
        self.means = np.array(param_grid["means"])
        self.covariances = np.array(param_grid["covariances"])
	self.covar = np.array(param_grid["covariances"][0])
        self.coef = np.array(param_grid["coef"])
	self.beta = param_grid["beta"]
	self.energy_data = np.array(param_grid["energy_data"])
	self.energy_cluster = np.array(param_grid["energy_cluster"])
		
	variance = 0.1
	
	self.energy_lambda = 1/float(variance)
	
        self.membership_probabilities = np.empty([len(self.data), len(self.coef)]) #evaluated log membership probabilities
        self.threshold = threshold
        self.iter = iterations
	self.reg_covar = reg_covar
	self.precisions_chol = np.empty(self.covariances.shape)
	self.log_determinant = []   

    def E_step(self):
	log_coefs = np.log(self.coef)
	probability = []
	for j in range(len(self.coef)):
		probability.append(prob_dis.log_pdf(self.data, self.means[j],self.covariances[j], self.beta, self.energy_data, self.energy_cluster[j], self.precisions_chol[j],self.log_determinant[j]))

	probability = np.array(probability)

	self.membership_probabilities = log_coefs.reshape((-1,1)) + probability
	
	max_vals = np.amax(self.membership_probabilities, axis=0)
	
	total = np.log(np.sum(np.exp(self.membership_probabilities - max_vals),axis=0)) + max_vals
	
	self.membership_probabilities -= total
	self.membership_probabilities = np.exp(self.membership_probabilities).T
		
        return
    
    def M_step(self):
        #updating the coefficients
	self.coef = np.sum(self.membership_probabilities, axis=0)
        self.coef /= float(len(self.data))

        #updating the covariances
	sum_denom = np.sum(self.membership_probabilities, axis=0)
        for j in range(len(self.covariances)):
	    diff = self.data - self.means[j]
	    
	    self.covariances[j]	= np.dot(self.membership_probabilities[:,j]*diff.T,diff)/sum_denom[j]
	    
	    #regularizing the covariance matrix
	    self.covariances[j] += self.reg_covar*np.eye(self.covariances[j].shape[0])
	  

	##updating beta
	sum_num = 0
	sum_denom = 0
        for i in range(len(self.energy_data)):
		for j in range(len(self.energy_cluster)):
			if self.energy_cluster[j] < self.energy_data[i]:	
				sum_num += self.membership_probabilities[i][j]
               			sum_denom += self.membership_probabilities[i][j]*(self.energy_data[i] - self.energy_cluster[j])
             
        self.beta = sum_num / float(sum_denom)
	#print "Beta: ", self.beta 

        return


    def calculate_log_probability(self):
        log_coefs = np.log(self.coef)
	probability = []
	for j in range(len(self.coef)):
		probability.append(prob_dis.log_pdf(self.data, self.means[j],self.covariances[j], self.beta, self.energy_data, self.energy_cluster[j], self.precisions_chol[j],self.log_determinant[j]))

	probability = np.array(probability).T

	#expectation_log_likelihood = np.sum((log_coefs + probability - np.log(self.membership_probabilities*1.0))*self.membership_probabilities)
	
	sum = 0
        for i in range(len(self.data)):
            temp_sum = 0
            for j in range(len(self.coef)):
                if self.membership_probabilities[i][j] >= 1e-6:
                    temp_sum += self.membership_probabilities[i][j]*(np.log(self.coef[j]) + probability[i][j] - np.log(float(self.membership_probabilities[i][j])))
	    sum += temp_sum

	#for j in range(len(self.coef)):
	#	sum += -0.5*(self.energy_lambda*self.energy_cluster[j]*self.energy_cluster[j]) - 0.5*(np.log(2*np.pi)) + np.log(np.sqrt(self.energy_lambda))

	#sum += -0.5*(self.energy_lambda*self.beta*self.beta) - 0.5*(np.log(2*np.pi)) + np.log(np.sqrt(self.energy_lambda))

        return sum

    def calculate_likelihood(self,data, energy_data):
        sums = 0
        for i in range(len(data)):
            temp_sum = 0
            for j in range(len(self.coef)):
                    temp_sum +=  self.coef[j]*np.exp(prob_dis.log_pdf(data[i], self.means[j],self.covariances[j], self.beta, energy_data[i], self.energy_cluster[j], self.precisions_chol[j], self.log_determinant[j]))

	    sums += np.log(temp_sum)
        return sums

    def fit(self):
        previous_iter = -1.0*float("inf")
        iterations = 0
	self.precisions_chol, self.log_determinant = prob_dis.compute_precisions_chol(self.covariances, self.reg_covar)
        while iterations < self.iter: 
	    print "E-Step",	
            #E-step
	    start = time.time()
            self.E_step()
	   
	    end = time.time()
	    print(end - start)
	
	    current_iter = self.calculate_log_probability()
            print current_iter, iterations,
	    print current_iter - previous_iter
            if (abs(current_iter - previous_iter) < self.threshold):
                break

	    print "M-Step",
            #M-step
	    start = time.time()
            self.M_step()
	    end = time.time()
	    print(end - start)

	    
	    self.precisions_chol,self.log_determinant = prob_dis.compute_precisions_chol(self.covariances, self.reg_covar)
	    
            
            iterations += 1
	    previous_iter = current_iter
	    print 

    def get_params(self):
        #dictionary containing parameter values
        d = shelve.open("EM_params_with_full_covariance")

        d["means"] = self.means
        d["variances"] = self.covariances
        d["coef"] = self.coef
	d["beta"] = self.beta 
	d["energy_cluster"] = self.energy_cluster 
        return d

