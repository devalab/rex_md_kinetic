import misc, operator
import auxillary_data_structures as aux
import numpy as np
import shelve, gc, joblib, sys, copy	
import matplotlib.pyplot as plt
import preprocessing
import expectation_maximization as em
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn import metrics
import prob_distributions as prob_dis
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy.optimize import linprog

def tsne(load=True,jump=9):
    #d = shelve.open("dynamic_clustering")
    X = preprocessing.load_residues('reduced_dimensions.pkl')

    #clusters = []
    #for i in range(0, X.shape[0], jump):
    #	clusters.append(int(d[str(i)]))
    X = X[::jump]
    d = {}
    gc.collect()
    #print X.shape, len(clusters)

    if load == False:
        model = TSNE(n_components=2, random_state=0, perplexity=40)
        y = model.fit_transform(X)
        joblib.dump(model,'tsne_model.pkl')
    else:
        clf = joblib.load('tsne_model.pkl')
        y = clf.embedding_

    #print y.shape
    #print np.array(clusters).shape

    """
    # Plot our dataset.
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    p = plt.scatter(y[:, 0], y[:, 1], c=np.array(clusters), cmap=plt.cm.rainbow)
    plt.colorbar(p)
    plt.legend()
    # plt.show()
    plt.savefig("TSNE_40.eps", format='eps', dpi=300)
    """   
    return

def density_clustering(jump=9, min_number_of_samples = 5):
    X = preprocessing.load_residues('reduced_dimensions.pkl')

    clf = joblib.load('tsne_model.pkl')
    y = clf.embedding_

    # determining value of eps - data is of uniform density
    neigh = NearestNeighbors(4)
    neigh.fit(y)

    plt.clf()
    distances = neigh.kneighbors()[0][:, 3]
    #print distances.shape
    plt.plot(np.arange(distances.shape[0]), np.array(sorted(distances)))

    # plt.plot(np.arange(distances.shape[0]), first_order_gradients[:])
    # print second_order_gradients.argmax()

    # print second_order_gradients.argmax()
    eps_cutoff = np.array(sorted(distances))[int(0.99 * distances.shape[0])]
    #print eps_cutoff, int(0.99 * distances.shape[0])
    # plt.show()

    model = DBSCAN(eps=eps_cutoff, min_samples=min_number_of_samples)
    model.fit_predict(y)
    joblib.dump(model, 'dbscan_model.pkl')
   #print set(model.labels_), len(set(model.labels_))

    """
    not_noise_indices = np.where(model.labels_ != -1)
    new_cluster_labels = model.labels_[not_noise_indices]
    new_ground_truth = (np.array(clusters))[not_noise_indices]
    #print new_cluster_labels.shape
    #print set(clusters), len(set(clusters))

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    p = plt.scatter(y[:, 0], y[:, 1], c=np.array(model.labels_), cmap=plt.cm.rainbow)
    plt.colorbar(p)
    # plt.show()
    plt.savefig("DBSCAN.eps", format='eps', dpi=300)

    labels_true = new_ground_truth
    labels_pred = new_cluster_labels
    print "AMI Score: ", metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    """
    return

def evaluate_metastable_states(jump):
    d = shelve.open("dynamic_clustering")
    X = preprocessing.load_residues('reduced_dimensions.pkl')
    clusters = []
    for i in range(0, X.shape[0], jump):
        clusters.append(int(d[str(i)]))

    model = joblib.load('dbscan_model.pkl')
    not_noise_indices = np.where(model.labels_ != -1)
    new_cluster_labels = model.labels_[not_noise_indices]
    new_ground_truth = (np.array(clusters))[not_noise_indices]

    labels_true = new_ground_truth
    labels_pred = new_cluster_labels
    return metrics.adjusted_mutual_info_score(labels_true, labels_pred)


def find_cluster_centres(jump=18, pdb_file="", dcd_pkl_filename="", load=False):
    # this function finds the most probable 2-D location of each cluster
    dbscan = joblib.load('dbscan_model.pkl')
    X = preprocessing.load_residues('reduced_dimensions.pkl')
    indices = dbscan.core_sample_indices_
    X = X[::jump]  # very important line jump as in t-SNE we skip frames according to the jump parameter
    #print X.shape
    if load == False:
        core_points = X[indices]
        #print core_points.shape
        labels = dbscan.labels_[indices]
        #print labels.shape
        cluster_frames_dict = {}
        for lbl in set(labels):
            cluster_frames_dict[lbl] = []

        for index in range(core_points.shape[0]):
            cluster_frames_dict[labels[index]].append(index)

        cluster_representative_point = {}
        for key in cluster_frames_dict.keys():
            cluster_representative_point[key] = misc.most_probable_structure_in_cluster(cluster_frames_dict[key], X, pdb_file, key, "tsne", dcd_pkl_filename, jump)

        joblib.dump(cluster_representative_point, "tsne_cluster_representative_point.pkl")

    else:
        cluster_representative_point = joblib.load("tsne_cluster_representative_point.pkl")

    # make_clusters
    cluster_representative_index = {}
    for key in cluster_representative_point.keys():
        frame_indices = misc.find_indices_of_clusters(dbscan.labels_, key)
	print "mol new rna.psf"
	for x in frame_indices:
   		print "mol addfile " + dcd_pkl_filename[:3] + "-recenter-solute.dcd first " + str(x*jump) + " last " + str(x*jump) +" waitfor all" 
   	print
        temp_index = misc.get_closest_structure_index(np.array(frame_indices), X, cluster_representative_point[key])
        cluster_representative_index[key] = temp_index
    return cluster_representative_point, cluster_representative_index


def get_transition_probabilities(cluster_representative_point, cluster_representative_index, temp, jump, iterations, train_model = True):
	data_points = preprocessing.load_residues('reduced_dimensions.pkl')
	energies = preprocessing.create_energy_matrix(get_from_file=1)
	
	energies = np.sum(energies,axis=1)
	dbscan = joblib.load('dbscan_model.pkl')
	beta_val = 1.0/(0.0019872041*temp)
	#EM to estimate parameters of mixture model
	
	if train_model == True:			
		init_cov = np.cov(data_points, rowvar = False)
		cov = []
		coefs = []
		means = []
		energies_cluster = []
		for key in cluster_representative_point.keys():
			frame_indices = misc.find_indices_of_clusters(dbscan.labels_, key)
			params = {'bandwidth': np.logspace(-1, 0, 20)}
			grid = GridSearchCV(KernelDensity(), params)
			grid.fit(energies[np.array(frame_indices)].reshape(-1,1))

			# use the best estimator to compute the kernel density estimate
			kde = grid.best_estimator_
			sampling_points = kde.sample(n_samples=10000, random_state=20)
			mean_energy = float(np.mean(sampling_points, axis=0))
			

			cov.append(init_cov)
			
			means.append(cluster_representative_point[key])
			energies_cluster.append(mean_energy)
			
			coefs.append(1.0/len(cluster_representative_point.keys()))
	
		#print energies_cluster
		param_grid = {"means": means, "data": data_points, "beta": beta_val, "covariances": cov, "energy_data": energies, "energy_cluster": energies_cluster, "coef": coefs}

		model = em.Expectation_Maximization(param_grid, threshold=1e-4, reg_covar=1e-6, iterations=iterations)
		model.fit()
		model.get_params() #saves params in shelve file names "EM_params"
		mixture_params = dict(shelve.open("EM_params_with_full_covariance"))
		
		joblib.dump(mixture_params, "EM_params.pkl")

	mixture_params = joblib.load("EM_params.pkl")
	#print mixture_params["beta"], mixture_params["coef"]
	log_transition_matrix = aux.Autovivification() #T_i_j represents prob of going from i to j
	
	for i in cluster_representative_point.keys():
		covar_matrix = mixture_params["variances"][i]
		prec_chol, log_det = prob_dis.compute_precisions_chol(np.array([covar_matrix]))
		for j in cluster_representative_point.keys():
			temp_indices = misc.find_indices_of_clusters(dbscan.labels_, j)			
		

			log_probabilites = prob_dis.log_pdf(data_points[::jump][np.array(temp_indices)], mixture_params["means"][i], covar_matrix, mixture_params["beta"], energies[::jump][np.array(temp_indices)], mixture_params["energy_cluster"][i], prec_chol[0], log_det[0])
	 		max_val = np.amax(log_probabilites)
			

			total = np.log(np.sum(np.exp(log_probabilites - max_val))) + max_val
			log_transition_matrix[i][j] = total
	
	return log_transition_matrix

def get_individual_transitions(cluster_representative_point, jump):
    transition_matrix = aux.Autovivification() #T_i_j represents prob of going from i to j
    mixture_params = shelve.open("EM_params_with_full_covariance")

    data_points = preprocessing.load_residues('reduced_dimensions.pkl')
    energies = preprocessing.create_energy_matrix(get_from_file=1)
	
    energies = np.sum(energies,axis=1)

    dbscan = joblib.load('dbscan_model.pkl')

    for i in cluster_representative_point.keys():
	#print "cluster_id:", i
        covar_matrix = mixture_params["variances"][i]
        prec_chol, log_det = prob_dis.compute_precisions_chol(np.array([covar_matrix]))
        
        parent_temp_indices = misc.find_indices_of_clusters(dbscan.labels_, i)    

        parent_temp_transition_probs = aux.Autovivification()    

        parent_log_probabilities = prob_dis.log_pdf(data_points[::jump][np.array(parent_temp_indices)], mixture_params["means"][i], covar_matrix, mixture_params["beta"], energies[::jump][np.array(parent_temp_indices)], mixture_params["energy_cluster"][i], prec_chol[0], log_det[0])

        for j in cluster_representative_point.keys():
            temp_indices = misc.find_indices_of_clusters(dbscan.labels_, j)            
        
            child_log_probabilities = prob_dis.log_pdf(data_points[::jump][np.array(temp_indices)], mixture_params["means"][i], covar_matrix, mixture_params["beta"], energies[::jump][np.array(temp_indices)], mixture_params["energy_cluster"][i], prec_chol[0], log_det[0])
             
            for index in range(parent_log_probabilities.shape[0]):
                max_val = np.maximum(child_log_probabilities, np.array([parent_log_probabilities[index]]))
        
                trans_prob = np.exp(child_log_probabilities - max_val - np.log(np.exp(child_log_probabilities - max_val) + np.exp(np.array([parent_log_probabilities[index]]) - max_val)))
                
                trans_prob[trans_prob == 0.5] = 0.0
                
            
                parent_temp_transition_probs[index][j] = copy.deepcopy(trans_prob)
        
        #normalize_probabilities_for_individual_structures
        for index in parent_temp_transition_probs.keys():
            denom = 0.0
            for j in parent_temp_transition_probs[index].keys():
                denom += np.sum(parent_temp_transition_probs[index][j])
            
            for j in parent_temp_transition_probs[index].keys():
                parent_temp_transition_probs[index][j] /= denom

	    	transition_matrix[parent_temp_indices[index]][j] = parent_temp_transition_probs[index][j]

    return transition_matrix

def make_entire_data_transition_matrix(tm, cluster_representative_point, start, end):
	transition_matrix = aux.Autovivification()
	dbscan = joblib.load('dbscan_model.pkl')
	
	par_1_indices = misc.find_indices_of_clusters(dbscan.labels_, start)
	start_index = np.argmax(par_1_indices)
	par_2_indices = misc.find_indices_of_clusters(dbscan.labels_, end)
	end_index = np.argmax(par_2_indices)

	start_node = par_1_indices[start_index]
	end_node = par_2_indices[end_index]
	
	#print start_node, end_node
	
	temp_indices = misc.find_indices_of_clusters(dbscan.labels_, start)
	for i in sorted(tm.keys()):
		if i in par_1_indices or i in par_2_indices:
			for index in range(len(temp_indices)):
				transition_matrix[i][temp_indices[index]] = np.log(tm[i][start][index]) 

	temp_indices = misc.find_indices_of_clusters(dbscan.labels_, end)
	for i in sorted(tm.keys()):
		if i in par_1_indices or i in par_2_indices:
			for index in range(len(temp_indices)):
				transition_matrix[i][temp_indices[index]] = np.log(tm[i][end][index])
 
	index_ = 0
	for i in sorted(transition_matrix.keys()):
		if start_node == i:
			si = index_

		if end_node == i:
			ei = index_

		index_ += 1

	#print si, ei
	return transition_matrix, start_node, end_node, si, ei

def get_commitor_probabilities(log_transition_matrix, start_state, end_state):
	trans_mat = []
	for i in sorted(log_transition_matrix.keys()):
		temp = []
		for j in sorted(log_transition_matrix[i].keys()):
			if i != j:
	    			temp.append(np.exp(log_transition_matrix[i][j]))
			else:
				temp.append(0.0)		
		trans_mat.append(temp)
	trans_mat = np.array(trans_mat)
	print trans_mat.shape, start_state, end_state
	b = trans_mat[:,end_state]
	
	b_new = b[:end_state]
	if end_state != trans_mat.shape[0] - 1:
		b_new = np.hstack((b_new, b[end_state+1:]))
	
	#remove entry of end_state from transition_matrix
	new_trans_mat_col = trans_mat[:,:end_state]
	if end_state != trans_mat.shape[1] - 1:
		new_trans_mat_col = np.hstack((new_trans_mat_col, trans_mat[:,end_state+1:]))


	new_trans_mat_row = new_trans_mat_col[:end_state,:]
	if end_state != new_trans_mat_col.shape[0] - 1:
		new_trans_mat_row = np.vstack((new_trans_mat_row, new_trans_mat_col[end_state+1:,:]))
	

	A = np.eye(new_trans_mat_row.shape[0]) - new_trans_mat_row
	p, l, u = scipy.linalg.lu(A, permute_l=False, overwrite_a=False, check_finite=True)
	b_permuted = np.dot(p.T,b_new)

	w = scipy.linalg.solve_triangular(l, b_permuted, lower=True)
	x = scipy.linalg.solve_triangular(u, w, lower=False)

	"""c = np.zeros((b_new.shape))
	bounds_tuple = ()
	for i in range(trans_mat.shape[0]):
		t = ((0,1),)
		if i == start_state:
			t = ((0,0),)
		if i == end_state:
			continue	
		bounds_tuple += t
	
	res = linprog(c, A_eq=(np.eye(new_trans_mat_row.shape[0]) - new_trans_mat_row), b_eq=b_new, bounds=bounds_tuple, options={"disp": True,'tol': 1e-16})
	x = res.x"""
	temp = np.dot(A, x)
	#print temp, b_new	

	commitor_probs = []
	index = 0
	for i in log_transition_matrix.keys():
		if i == end_state:
			commitor_probs.append(1.0)
		else:
			commitor_probs.append(x[index])
			index += 1	
	
	return commitor_probs

def get_commitors(log_transition_matrix, start_state, end_state):
	forward_commitors = get_commitor_probabilities(log_transition_matrix, start_state, end_state)
	print forward_commitors
	
	backward_commitors = get_commitor_probabilities(log_transition_matrix, end_state, start_state)
	print backward_commitors
	#sys.exit(0)
	return forward_commitors, backward_commitors

def get_flux(forward_commitors, backward_commitors, log_transition_matrix, equilibrium_distribution):
	mixture_params = shelve.open("EM_params_with_full_covariance")
	log_coefs = np.log(mixture_params["coef"])
	flux_matrix = aux.Autovivification()	
	for i in log_transition_matrix.keys():
		for j in log_transition_matrix[i].keys():
			forward_flux = np.log(equilibrium_distribution[i]) + log_transition_matrix[i][j] + np.log(forward_commitors[j]) + np.log(backward_commitors[i])
			backward_flux = np.log(equilibrium_distribution[j]) + log_transition_matrix[j][i] + np.log(forward_commitors[i]) + np.log(backward_commitors[j])	
			flux_matrix[i][j] = max(0,np.exp(forward_flux) - np.exp(backward_flux))
			
	return flux_matrix

def get_most_probable_path(frame_cluster_ids):
    # cluster ids is the dynamic cluster dict for each mean cluster frames is the gaussian cluster each frame belongs to
    sequence = []
    for i in range(len(frame_cluster_ids)):
        if (i >= 1):
            if sequence[-1] != frame_cluster_ids[i] and frame_cluster_ids[i] != -1:
                sequence.append(frame_cluster_ids[i])
        else:
            sequence.append(frame_cluster_ids[i])
    return sequence


def construct_graph(cluster_representative_point,transition_matrix):
    cluster_membership = {}
    dbscan = joblib.load('dbscan_model.pkl')
    #print dbscan.labels_.shape
    for i in range(dbscan.labels_.shape[0]):
        try:
            cluster_membership[dbscan.labels_[i]] += 1

        except:
            cluster_membership[dbscan.labels_[i]] = 1

    for i in cluster_membership:
        cluster_membership[i] = cluster_membership[i] / (dbscan.core_sample_indices_.shape[0] * 1.0)  ###probability

    vertices = []
    for i in sorted(cluster_representative_point.keys()):
        temp = []
        index = 0
        for j in sorted(cluster_representative_point.keys()):
            #dist = misc.distance(cluster_representative_point[i], cluster_representative_point[j])
            dist = -1*transition_matrix[i][j]
	    temp.append(float(dist))

            index += 1
        vertices.append(temp)
    return np.array(vertices)


def check_adjacent_edges(i, j, adjacency_list, modified_graph):
    total_edge = 0
    num_of_edge = 0
    for jj in range(len(adjacency_list)):
        if jj != j:
            if modified_graph[i][jj] != float("inf"):
                total_edge += adjacency_list[i][jj]
                num_of_edge += 1

    for ii in range(len(adjacency_list)):
        if ii != i:
            if modified_graph[ii][j] != float("inf"):
                total_edge += adjacency_list[ii][j]
                num_of_edge += 1

    avg_length = total_edge / float(num_of_edge)

    if adjacency_list[i][j] > avg_length:
        return False
    else:
        return True

def mst(adjacency_list):
	modified_graph = aux.Autovivification()	
	mst_set = aux.disjoint(np.arange(len(adjacency_list)))
	distances = {}
	for i in range(len(adjacency_list)):        	
        	for j in range(len(adjacency_list)):	
			modified_graph[i][j] = float("inf")
        	    	distances[(i,j)] = adjacency_list[i][j]

	#sort edges according to weights
	#print sorted(distances.items(), key=operator.itemgetter(0), reverse=True)
	for jj in sorted(distances.items(), key=operator.itemgetter(1)):
		nodes = jj[0]
		ipar = mst_set.findparent(nodes[0])
        	jpar = mst_set.findparent(nodes[1])
		if ipar != jpar:
			modified_graph[nodes[0]][nodes[1]] = jj[1]
			modified_graph[nodes[1]][nodes[0]] = jj[1]
			mst_set.union(nodes[0],nodes[1])

	return modified_graph

def construct_modular_graph(adjacency_list):
    # use modified version of graph_clustering
    modified_graph = aux.Autovivification()
    for i in range(len(adjacency_list)):
        distances = {}
        for j in range(len(adjacency_list)):
            distances[adjacency_list[i][j]] = j

        for jj in sorted(distances.keys(), reverse=True):
            j = distances[jj]
            if i != j:
                if modified_graph[i][j] != float("inf") and modified_graph[j][i] != float("inf"):
                    Is_Present = check_adjacent_edges(i, j, adjacency_list, modified_graph)
                    # if Is_Present == True:
                    #	modified_graph[i][j] = adjacency_list[i][j]
                    # else:

                    modified_graph[i][j] = float("inf")
                    modified_graph[j][i] = float("inf")

    # draw_graph(adjacency_list, modified_graph)
    return modified_graph


def number_of_edges_present(Graph_As_Dict):
    counter = 0
    total = 0
    modified_graph = Graph_As_Dict
    for i in modified_graph.keys():
        for j in modified_graph.keys():
            if modified_graph[i][j] != float("inf"):
                counter += 1
            total += 1

    print "Number of edges retained out of total:", counter, "/", total


def dfs(modified_graph, key, visited, scc, low, discovered, counter, stack, in_stack):
    if key in visited:
        return visited, scc, low, discovered, counter, stack, in_stack
    visited[key] = True
    stack.append(key)
    in_stack[key] = True
    low[key] = discovered[key] = counter
    counter += 1
    for vertex in modified_graph.keys():
        if modified_graph[key][vertex] != float("inf") and key != vertex:
            if vertex in visited:
                if vertex in in_stack and in_stack[vertex] == True:
                    low[key] = min(low[key], discovered[vertex])
            else:
                visited, scc, low, discovered, counter, stack, in_stack = dfs(modified_graph, vertex, visited, scc, low,
                                                                              discovered, counter, stack, in_stack)
                low[key] = min(low[key], low[vertex])
                # scc.union(key,vertex)
    top_element = stack[-1]

    # print key, low[key], discovered[key]
    if low[key] == discovered[key]:
        # print stack
        while (len(stack) > 0):
            top_element = stack[-1]

            scc.union(key, top_element)

            in_stack[top_element] = False
            stack.pop()
            if top_element == key:
                break
    return visited, scc, low, discovered, counter, stack, in_stack


def find_scc(modified_graph):
    scc = aux.disjoint(modified_graph.keys())
    visited = {}
    low = {}
    discovered = {}
    stack = []
    in_stack = {}
    counter = 0
    for key in modified_graph.keys():
        visited, scc, low, discovered, counter, stack, in_stack = dfs(modified_graph, key, visited, scc, low,
                                                                      discovered, counter, stack, in_stack)
    scc.compress()

    return scc


def get_min_distance_between_components(A, B, adjacency_list, modified_graph):
    min_dist = float("inf")
    pair = ()

    for i in A:
        for j in B:
            # print i, j, adjacency_list[i][j]
            if modified_graph[i][j] != float("inf"):
                return -1, pair
            elif min_dist > adjacency_list[i][j]:
                min_dist = min(min_dist, adjacency_list[i][j])
                pair = (i, j)

    return min_dist, pair


def make_graph_complete(modified_graph, adjacency_list, scc, start):
    components = {}
    index = 0
    for i in scc.parent:
        temp = scc.get_components(i)
        if len(temp) > 0:
            if start in temp:
                index = i
            components[i] = temp
    print "Number of components =", len(components.keys())
    print components
    if len(components.keys()) == 1:
        return modified_graph, True
    else:
        min_dist = float("inf")
        min_pair = (index, index)
        nearest_component = index

        # find the nearest_component
        for j in components:
            # print index, j
            if index != j:
                new_dist, pair = get_min_distance_between_components(components[index], components[j], adjacency_list,
                                                                     modified_graph)
                if new_dist < min_dist and new_dist != -1:
                    min_dist = min(min_dist, new_dist)
                    min_pair = pair
                    nearest_component = j
        # print nearest_component
        # print min_dist, min_pair

        # path from index to j
        if min_dist != -1 and min_dist != float("inf"):
            modified_graph[min_pair[0]][min_pair[1]] = min_dist

        # path from j to index
        min_pair = (index, index)
        min_dist, min_pair = get_min_distance_between_components(components[nearest_component], components[index],
                                                                 adjacency_list, modified_graph)
        # print min_dist, min_pair
        if min_dist != -1:
            modified_graph[min_pair[0]][min_pair[1]] = min_dist
        return modified_graph, False

def map_PES():
	dbscan = joblib.load('dbscan_model.pkl')
	labels = dbscan.labels_
	unique = {}
	order_of_discovery = []
	for i in labels:
		if i in unique.keys():
			continue
		else:
			unique[i] = 1
			if i != -1:
				order_of_discovery.append(i)
		
	return order_of_discovery
	
	
def get_start_and_end_times_of_metastable_states(): 
    dbscan = joblib.load('dbscan_model.pkl')
    labels = dbscan.labels_
    sequence = []
    metastable_states = list(set(labels))
    dwell_probability = {}
    for i in range(len(labels)):
        if (i >= 1):
            if sequence[-1] != labels[i]:
		dwell_probability[sequence[-1]].append(i - 1)
		try:
			dwell_probability[labels[i]].append(i)
		except KeyError:
			dwell_probability[labels[i]] = [i]
	
                sequence.append(labels[i])
		

        else:
            sequence.append(labels[i])
	    dwell_probability[labels[i]] = [i]
    dwell_probability[sequence[-1]].append(i)
    start_and_end_times = {}
    for key in dwell_probability.keys():
		index = 0
		while index < (len(dwell_probability[key])):
			try:
				start_and_end_times[key].append([dwell_probability[key][index],dwell_probability[key][index+1]])
			except KeyError:
				start_and_end_times[key] = [[dwell_probability[key][index],dwell_probability[key][index+1]]]
			index = index + 2


    return start_and_end_times

	






