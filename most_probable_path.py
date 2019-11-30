import joblib, operator
import mdtraj as md
from collections import OrderedDict
import preprocessing, misc
import auxillary_data_structures as ds
import graph_based_unfolding as gbu
from sklearn.cluster import KMeans
import math, shelve, copy, scipy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
import sys
import matplotlib.pyplot as plt

def cluster_trajectory_kmeans(fit=True):
    if fit == True:
        X = preprocessing.load_residues('reduced_dimensions.pkl')
        model = KMeans(n_clusters=1000)
        model.fit(X)
        joblib.dump(model, "KMEANS.pkl")

    else:
        model = joblib.load("KMEANS.pkl")
        X = preprocessing.load_residues('reduced_dimensions.pkl')

        mean_cluster_ids = shelve.open("kmeans_trajectory_clustering")  # this structure stores cluster-ids for each frame of the trajectory
        #print X.shape
        for i in range(X.shape[0]):
            prediction = model.predict(X[i].reshape((1, -1)))
            mean_cluster_ids[str(i)] = int(prediction)

        number_of_current_clusters = model.cluster_centers_.shape[0]
        d = dict(mean_cluster_ids)
        mean_cluster_ids.close()


def dfs_markov(vertex, dynamic_cluster_ids, transition_matrix, dynamic_clusters, visited, meta_stability_criteria):
    # print visited
    model = joblib.load("KMEANS.pkl")
    i = vertex
    max_prob = 0.0
    index = i

    if vertex in visited:
        return visited
    visited[vertex] = 1

    if transition_matrix[i][i] >= meta_stability_criteria:
        index = i

    else:
        for j in dynamic_cluster_ids:
            if j != i and transition_matrix[i][j] > max_prob:
                max_prob = transition_matrix[i][j]
                index = j

        # finding closest cluster
        min_dist = float("inf")
        for j in dynamic_cluster_ids:
            if transition_matrix[i][j] == max_prob:
                temp_dist = model.cluster_centers_[i] - model.cluster_centers_[j]
                temp_dist = np.dot(temp_dist.T, temp_dist)
                if min_dist <= temp_dist:
                    index = j
                    min_dist = temp_dist

    dynamic_clusters.union(i, index)
    dfs_markov(index, dynamic_cluster_ids, transition_matrix, dynamic_clusters, visited, meta_stability_criteria)
    return visited


def get_most_probable_path(frame_cluster_ids):
    # cluster ids is the dynamic cluster dict for each mean cluster frames is the gaussian cluster each frame belongs to
    sequence = []
    for i in range(len(frame_cluster_ids)):
        if (i >= 1):
            if sequence[-1] != frame_cluster_ids[str(i)]:
                sequence.append(frame_cluster_ids[str(i)])
        else:
            sequence.append(frame_cluster_ids[str(i)])

    return sequence

def core_clusters(dynamic_clustering, pdb_file, dcd_pkl_filename):
    # dict of trajectory frames
    dynamic_clustering_frames_list = {}  # dict of [cluster_no - frameids]

    for i in dynamic_clustering:
        try:
            dynamic_clustering_frames_list[dynamic_clustering[i]].append(int(i))
        except:
            dynamic_clustering_frames_list[dynamic_clustering[i]] = [int(i)]

    frames = preprocessing.load_residues('reduced_dimensions.pkl')
    distances_of_frames_in_cluster = {}  # this is distance-frameindex mapping
    avg_structure_in_cluster = {}  # this is distance-frameindex mapping

    for i in dynamic_clustering_frames_list:
        #print i, "cluster_id"
        temp = misc.most_probable_structure_in_cluster(dynamic_clustering_frames_list[i], frames, pdb_file, i, "dynamic", dcd_pkl_filename)
        total_number_of_strucutres = len(dynamic_clustering_frames_list[i])

        for j in range(total_number_of_strucutres):
            distances_of_frames_in_cluster[misc.distance(frames[dynamic_clustering_frames_list[i][j]], temp)] = dynamic_clustering_frames_list[i][j]

        distance_value = sorted(distances_of_frames_in_cluster.keys())[(total_number_of_strucutres / 2) + 1]
        distances_of_frames_in_cluster[i] = distance_value
        avg_structure_in_cluster[i] = temp
    return distances_of_frames_in_cluster, avg_structure_in_cluster


def dynamic_cluster_trajectory(meta_stability_criteria = 0.9, pdb_file = "", dcd_pkl_filename = ""):
    model = joblib.load("KMEANS.pkl")
    X = preprocessing.load_residues('reduced_dimensions.pkl')
    #print X.shape

    mean_cluster_ids = shelve.open("kmeans_trajectory_clustering")  # this structure stores cluster-ids for each frame of the trajectory
    if (len(mean_cluster_ids) == 0):
        for i in range(X.shape[0]):
            prediction = model.predict(X[i].reshape((1, -1)))
            mean_cluster_ids[str(i)] = int(prediction)
    #print "yo"
    number_of_current_clusters = model.cluster_centers_.shape[0]
    d = dict(mean_cluster_ids)
    mean_cluster_ids.close()
    while (True):
        cluster_membership = {}
        #print number_of_current_clusters, "--"
        for i in range(X.shape[0]):
            try:
                cluster_membership[int(d[str(i)])] += 1

            except:
                cluster_membership[int(d[str(i)])] = 1

        transition_matrix = ds.Autovivification()

        for i in set(d.values()):
            for j in set(d.values()):
                transition_matrix[i][j] = 0

        for i in range(X.shape[0] - 1):
            transition_matrix[int(d[str(i)])][int(d[str(i + 1)])] += 1

        # normalizing values row-wise
        cluster_probability = {}
        for i in set(d.values()):
            sums = 0
            for j in set(d.values()):
                sums += transition_matrix[i][j]
            cluster_probability[i] = sums
            for j in set(d.values()):
                transition_matrix[i][j] /= sums * 1.0

        dynamic_clusters = ds.disjoint(set(d.values()))

        visited = {}
        for i in set(d.values()):
            temp_visited = dfs_markov(i, set(d.values()), transition_matrix, dynamic_clusters, visited, meta_stability_criteria)
            visited = copy.deepcopy(temp_visited)

        dynamic_clusters.compress()
        dynamic_clusters.save_structure()
        cluster_ids = dynamic_clusters.get_clusters(cluster_probability)
        new_clusters = cluster_ids.values()

        #print len(set(new_clusters))
        if number_of_current_clusters == len(set(new_clusters)):
            # applying cluster core correction
            distances_of_frames_in_cluster, avg_structure_in_cluster = core_clusters(d, pdb_file, dcd_pkl_filename)
            #print len(set(d.values())), "original"
            X = preprocessing.load_residues('reduced_dimensions.pkl')
            d_new = {}
            d_new["0"] = d["0"]
            for index in range(1, len(X)):
                dist = misc.distance(X[index], avg_structure_in_cluster[d[str(index)]])
                if dist > distances_of_frames_in_cluster[d[str(index)]]:
                    d_new[str(index)] = d_new[str(index - 1)]
                else:
                    d_new[str(index)] = d[str(index)]

            d = d_new
            dynamic_clustering = shelve.open("dynamic_clustering")
            dynamic_clustering.clear()
            for i in d:
                dynamic_clustering[i] = d[i]
            dynamic_clustering.close()
            #print len(set(d.values())), "original"
            sequence = get_most_probable_path(d)
            #print len(sequence)
            #print sequence
            return sequence, transition_matrix

        else:
            #print set(new_clusters)
            number_of_current_clusters = len(set(new_clusters))

            d_new = {}
            for i in range(X.shape[0]):
                d_new[str(i)] = cluster_ids[int(d[str(i)])]

            d = d_new
            dynamic_clustering = shelve.open("dynamic_clustering")
            dynamic_clustering.clear()
            for i in d:
                dynamic_clustering[i] = d[i]

            dynamic_clustering.close()

    return


def get_dynamic_cluster_sequence():
    frames = preprocessing.load_residues('reduced_dimensions.pkl')
    #print frames.shape
    d = shelve.open("dynamic_clustering")
    #print set(d.values())
    cluster_membership = {}
    for i in range(frames.shape[0]):
        try:
            cluster_membership[int(d[str(i)])] += 1

        except:
            cluster_membership[int(d[str(i)])] = 1

    transition_matrix = ds.Autovivification()

    for i in set(d.values()):
        for j in set(d.values()):
            transition_matrix[i][j] = 0

    for i in range(frames.shape[0] - 1):
        transition_matrix[int(d[str(i)])][int(d[str(i + 1)])] += 1

    # normalizing values row-wise
    cluster_probability = {}
    for i in set(d.values()):
        sums = 0
        for j in set(d.values()):
            sums += transition_matrix[i][j]
        cluster_probability[i] = sums
        for j in set(d.values()):
            transition_matrix[i][j] /= sums * 1.0

    sequence = get_most_probable_path(d)
    return sequence, transition_matrix


def equilibrium_distribution(transition_matrix, is_log = False):
    trans_mat = []
    for i in transition_matrix.keys():
        temp = []
        for j in transition_matrix.keys():
            temp.append(transition_matrix[i][j])
	if is_log == True:
		trans_mat.append(np.exp(np.array(temp)))
	else:
        	trans_mat.append(temp)
    trans_mat = np.array(trans_mat)

    result = scipy.linalg.eig(trans_mat, left=True)
    unit_eigenvalue_index = 0
    for i in range(result[0].shape[0]):
        if abs(result[0][i] - (1.0 + 0.j)) < 1e-12:
            unit_eigenvalue_index = i
            break
    equilibrium_distribution = result[1][:, unit_eigenvalue_index] / np.sum(result[1][:, unit_eigenvalue_index])

    index = 0
    distribution = {}
    for i in transition_matrix.keys():
        print i, ":", np.round(equilibrium_distribution[index], 5).real
        distribution[i] = equilibrium_distribution[index]
        index += 1

    #print np.dot(equilibrium_distribution,trans_mat)
    return distribution


def construct_transition_graph(sequence, transition_matrix, cluster_probabilities):
    #print sequence
    vertices = list(set(sequence))
    edges = []
    lookup = ds.Autovivification()
    for i in range(len(sequence) - 1):
        try:
            lookup[sequence[i]][sequence[i + 1]] += 1
        except:
            edges.append((sequence[i], sequence[i + 1]))
            lookup[sequence[i]][sequence[i + 1]] = 1

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    colors = []
    for i in G.edges():
        colors.append(transition_matrix[i[0]][i[1]])

    color_node = []
    core_probability = {}
    index = 0
    for i in G.nodes():
        core_probability[i] = cluster_probabilities[i]

    color_nodes = []
    for i in G.nodes():
        color_nodes.append(core_probability[i])

    nx.draw(G, node_color=color_nodes, cmap=plt.cm.Reds, edge_color=colors, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("simple_path.png")  # save as png
    # plt.show() # display

    return vertices, edges


def get_most_probable_path_in_markov_chain(transition_matrix, start, end, is_log = False):
    #print start
    for i in transition_matrix.keys():
        for j in transition_matrix[i].keys():
		if is_log == False:	
			if transition_matrix[i][j] != 0:
               			transition_matrix[i][j] = 1 * math.log(transition_matrix[i][j], math.exp(1))
		 	else:
              			transition_matrix[i][j] = -1.0*float("inf")
		elif is_log == True:
			transition_matrix[i][j] = 1*transition_matrix[i][j]
          	#print transition_matrix[i][j]

    # using ordered dict is important as the method shortest_path (Djkstra) takes in an adjacency list representation of a graph where vertices are numbered from 1 to n, so for a consistent mapping with thhe keys (clusters_ids) of the transition_matrix it's imperitive to have some sort of consistent ordereing.

    transition_matrix_ordered = OrderedDict()
    transition_matrix_ordered = transition_matrix
    transition_matrix = sorted(transition_matrix_ordered.items(), key=lambda t: t[0])

    serial_mapping_of_clusters = {}
    reverse_mapping_of_serial_numbers = []
    vertices = []
    counter = 0
    for i in transition_matrix:
        # print i[0] #transition matrix is tuples with first element being the vertex and second element being a dict of neighbouring vertex probabilities
        serial_mapping_of_clusters[i[0]] = counter
        reverse_mapping_of_serial_numbers.append(i[0])
        counter += 1
        temp = []
        tempdict = OrderedDict()
        tempdict = i[1]
        temp_ordered = sorted(tempdict.items(), key=lambda t: t[0])
        for j in temp_ordered:
            # print j[0], j[1]
            temp.append(j[1])

        vertices.append(temp)
    # print reverse_mapping_of_serial_numbers, serial_mapping_of_clusters
   
    path = misc.djkstra_widest_path(np.array(vertices), serial_mapping_of_clusters[start], serial_mapping_of_clusters[end])

    for i in range(len(path)):
        path[i] = reverse_mapping_of_serial_numbers[path[i]]
    return path


def get_path_probability(transition_matrix, path, cluster_representative_index, jump):
    d = shelve.open("dynamic_clustering")
    frames = preprocessing.load_residues('reduced_dimensions.pkl')
    cluster_membership = {}
    for i in range(frames.shape[0]):
        try:
            cluster_membership[int(d[str(i)])] += 1

        except:
            cluster_membership[int(d[str(i)])] = 1

    initial_val = cluster_membership[int(d[str(0)])] / float(frames.shape[0])
    converted_clusters = [d[str((cluster_representative_index[0]) * jump)]]
    for i in range(1, len(path)):
        first = d[str((cluster_representative_index[path[i - 1]]) * jump)]
        second = d[str((cluster_representative_index[path[i]]) * jump)]
        #print first, second, transition_matrix[first][second]
        initial_val *= transition_matrix[first][second]

        #ensuring the path has unique clusters only
        next_state = d[str((cluster_representative_index[path[i]]) * jump)]
        if len(converted_clusters) != 0 and next_state != converted_clusters[-1]:
            converted_clusters.append(next_state)
    return converted_clusters, initial_val



def get_start_and_end_times_of_metastable_states():
    dynamic_clustering = shelve.open("dynamic_clustering")
    sequence = []
    metastable_states = list(set(dynamic_clustering.values()))
    dwell_probability = {}
    for i in range(len(dynamic_clustering)):
        if (i >= 1):
            if sequence[-1] != dynamic_clustering[str(i)]:
		dwell_probability[sequence[-1]].append(i - 1)
		try:
			dwell_probability[dynamic_clustering[str(i)]].append(i)
		except KeyError:
			dwell_probability[dynamic_clustering[str(i)]] = [i]
	
                sequence.append(dynamic_clustering[str(i)])
		

        else:
            sequence.append(dynamic_clustering[str(i)])
	    dwell_probability[dynamic_clustering[str(i)]] = [i]
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

def check_within_limits(point, start, end):
	if point >= start and point <= end:
		return True
	else:
		return False

def check_after_limits(point, start, end):
	if point > end:
		return True
	else:
		return False

def check_before_limits(point, start, end):
	if point < start:
		return True
	else:
		return False

def get_population_probabilities(number_of_frames, type_of_clustering="dynamic"):
    if type_of_clustering == "dynamic":
    	start_and_end_times = get_start_and_end_times_of_metastable_states()

    elif type_of_clustering == "graph":
	start_and_end_times = gbu.get_start_and_end_times_of_metastable_states()
	dbscan = joblib.load('dbscan_model.pkl')
	number_of_frames = dbscan.labels_.shape[0]

    time_dependent_population_probability = {}
    for key in start_and_end_times.keys():
	
	time_dependent_population_probability[key] = [0]*number_of_frames
	denominator = [0]*number_of_frames
	total = 0
	start_flag = True
	for index in range(len(start_and_end_times[key])):
		pair_1 = start_and_end_times[key][index]
		total += pair_1[1] - pair_1[0] + 1

		i = 0
		end = pair_1[1] - pair_1[0]
		while(i <= end):
			time_dependent_population_probability[key][i] += (end - i)
			i += 1
		
		"""for index_2 in range(index + 1, len(start_and_end_times[key])):
			pair_2 = start_and_end_times[key][index_2]
			start = pair_1[0]
			
			i = pair_2[0] - start
			end = pair_2[1] - start
			
			while(i <= end):
				time_dependent_population_probability[key][i] += max(0,(end - i + 1))
				pointer = pair_2[1] - i
				difference = max(0, pointer - pair_1[1])
				time_dependent_population_probability[key][i] -= difference
				i += 1
			
			start += 1
			while start <= pair_1[1]:
				i = pair_2[0] - start
				end = pair_2[1] - start
				time_dependent_population_probability[key][i] += max(0,(end - i + 1))
				pointer = pair_2[1] - i
				difference = max(0, pointer - pair_1[1])
				time_dependent_population_probability[key][i] -= difference
				start += 1"""

	for time_lag in range(1,number_of_frames):
		point = number_of_frames - 1 - time_lag	
		difference = 0
		for index in range(len(start_and_end_times[key])):
			pair = start_and_end_times[key][index]
			
			if check_after_limits(point, pair[0], pair[1]):
				break

			if check_before_limits(point, pair[0], pair[1]):
				difference += pair[1] - pair[0] - 1 	

			if check_within_limits(point, pair[0], pair[1]):
				difference += pair[1] - point - 1 

		denominator[time_lag - 1] = total - difference 

	for iterator in range(len(time_dependent_population_probability[key])):
		if time_dependent_population_probability[key][iterator] != 0:
			try:
				time_dependent_population_probability[key][iterator] /= (denominator[iterator]*1.0)	
			except:
				time_dependent_population_probability[key][iterator], (denominator[iterator]*1.0), iterator, key
	
    return time_dependent_population_probability

def get_sasa(cluster_indices, type_of_clustering):
    sasa_of_states = {}
    max_sasa = 0.0
    for index in cluster_indices:
       	f = md.load_pdb(type_of_clustering + "_" + str(index) + ".pdb")
        sasa = md.shrake_rupley(f)
	total_sasa = sasa.sum(axis=1)
        sasa_of_states[index] = float(total_sasa)
    return sasa_of_states

def survival_time_of_metastable_states(number_of_frames,type_of_clustering="dynamic"):
	time_dependent_population_probability = get_population_probabilities(number_of_frames,type_of_clustering)
	print "Number of Metastable_states: ", len(time_dependent_population_probability.keys())
	survival_times = {}
	for key in time_dependent_population_probability.keys():
		
		name = type_of_clustering + "_Chapman-Kolmogorov_Equation_Test_" + str(key) + ".eps"
		y_val = np.array(time_dependent_population_probability[key][:500])
		x_val = np.array(range(1,len(y_val) + 1))
	
		popt, pcov = curve_fit(misc.exp_func, x_val, y_val)
		y_func = misc.exp_func(x_val, *popt)
		
		plt.clf()
		plt.semilogy(x_val, y_val, label= "numerical estimation")
		plt.semilogy(x_val, y_func, "--", label='exponential_function')
		plt.xlabel('time (in ns)')
		plt.ylabel('p_i(t)')
		plt.legend()
		
		
		survival_times[key] = 1.0/popt[0]
		plt.savefig(name,format='eps',dpi=300)
		
		
	plt.clf()
	fig, ax = plt.subplots()
	metastable_state_label = []
	metastable_state_lifetimes = []
	
	sasa_vals = get_sasa(list(set(survival_times.keys())),"dynamic")
	sorted_sasa_tuples = sorted(sasa_vals.items(), key=operator.itemgetter(1))
	
	index = 0
	for key in survival_times.keys():
		metastable_state_label.append(sorted_sasa_tuples[index][0])
		metastable_state_lifetimes.append(survival_times[sorted_sasa_tuples[index][0]]/1000.0)
		index += 1
		
	ax.bar(np.arange(len(survival_times.keys())), np.array(metastable_state_lifetimes), color='#1976d2')
	plt.xlabel('state number')
	plt.ylabel('mean lifetime (in ns)')
	#ax.set_xticklabels(metastable_state_label)
	fig.savefig("metastable_state_mean_lifetime.eps", format='eps', dpi=300)	
	

	





	




