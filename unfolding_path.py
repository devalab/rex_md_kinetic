import preprocessing as pp
import most_probable_path as mpp
import misc
import sys, shelve, joblib, time, copy
import numpy as np
import graph_based_unfolding as gbu

def markov_chain(dcd, pdb, filename):
	#mpp.cluster_trajectory_kmeans(fit = True)
	#mpp.cluster_trajectory_kmeans(fit = False)
	#mpp.dynamic_cluster_trajectory(meta_stability_criteria = 0.93, pdb_file=pdb, dcd_pkl_filename=filename)
	
	sequence, transition_matrix = mpp.get_dynamic_cluster_sequence()
	#print len(sequence)
	distribution = mpp.equilibrium_distribution(transition_matrix)
	mpp.construct_transition_graph(sequence,transition_matrix,distribution)
	sequence, transition_matrix = mpp.get_dynamic_cluster_sequence()
	dynamic_clustering = shelve.open("dynamic_clustering")
	
	start, end = misc.get_cluster_ids_for_start_and_end(dynamic_clustering["0"],set(dynamic_clustering.values()), "dynamic")
	#print start, end
	path = mpp.get_most_probable_path_in_markov_chain(transition_matrix, start, end)
	print path
	dcd_array = pp.load_residues(filename)
	misc.write_dcd(dcd_array,path,"dynamic")
	misc.write_pdb(dcd_array,path,"dynamic")
	
	dcd_array = pp.load_residues(filename)
	return

def graph_based_method(dcd, pdb, filename,jump,min_number_of_samples):
	gbu.tsne(False,jump=jump)
	gbu.density_clustering(jump, min_number_of_samples)
	cluster_representative_point, cluster_representative_index = gbu.find_cluster_centres(jump, pdb, filename, False)
	
	log_transition_matrix = gbu.get_transition_probabilities(cluster_representative_point, cluster_representative_index, temp=411, jump=jump, iterations=1000, train_model = False)

	dbscan = joblib.load('dbscan_model.pkl')
	start = dbscan.labels_[0]
	
	sequence = gbu.get_most_probable_path(dbscan.labels_)	
	#print sequence
	#mpp.construct_transition_graph(sequence,transition_matrix,distribution)#

	start, end = misc.get_cluster_ids_for_start_and_end(start, set(dbscan.labels_), "tsne")
	#print start, end

	path = mpp.get_most_probable_path_in_markov_chain(copy.deepcopy(log_transition_matrix), start, end, is_log = True)
	#print path
	
	dcd_array = pp.load_residues(filename)
	misc.write_dcd(dcd_array,path,"tsne")
	misc.write_pdb(dcd_array,path,"tsne")

	print "The most reactive path is saved in your current working directory as 'tsne_unfolded_traj.dcd'"
	
	"""
	#compare the results to the MPP algorthm
	sequence, transition_matrix = mpp.get_dynamic_cluster_sequence()
	dynamic_clusters_path, path_prob = mpp.get_path_probability(copy.deepcopy(transition_matrix), path,cluster_representative_index,jump=jump)
	#print path_prob
	print "Graph:", (dynamic_clusters_path)

	path = mpp.get_most_probable_path_in_markov_chain(copy.deepcopy(transition_matrix), dynamic_clusters_path[0], dynamic_clusters_path[-1])
	print "Markov Chain", path
	misc.write_pdb(dcd_array,path,"dynamic")	
	"""
	return


def get_metastable_states(filename):
	path = gbu.map_PES()
	dcd_array = pp.load_residues(filename)
	misc.write_dcd(dcd_array,path,"tsne")

def preprocessing_data(dcd, pdb, filename):
	frames = pp.read_trajectory(dcd, pdb)
	print frames.shape
	pp.write_residues(frames, filename)
	
	#pp.reduce_dimensions(frames)
	frames = pp.create_energy_matrix(get_from_file=1)
	print frames.shape
	frames = pp.normalize_energy_matrix(frames, get_from_file=False)
	pp.reduce_dimensions(frames)
	pp.reduce_dimensions(frames, fit=False)
	print frames.shape

def choose_metastability_criteria(pdb, filename, jump):
	metastability = 0.92
	prev_quality = 0.605726655576
	while(metastability <= 1.0):
		old_time = time.time()
		mpp.dynamic_cluster_trajectory(meta_stability_criteria = metastability, pdb_file=pdb, dcd_pkl_filename=filename)
		print "Time Elapsed:", time.time() - old_time
		quality = gbu.evaluate_metastable_states(jump)
		if quality > prev_quality:
			prev_quality = quality
			best_val = metastability
			print best_val, prev_quality
		metastability += 0.01
	return best_val

def main():
	dcd = sys.argv[1] #the dcd file to analyze	
	pdb = sys.argv[2] #the pdb file of the initial structure
	filename = sys.argv[3] #name of the pkl file where mdtraj saves the whole dcd trajectory
	
	#Preprocessing
	#preprocessing_data(dcd, pdb, filename)

	#Markov_Chains_Method
	#markov_chain(dcd, pdb, filename)
	
	#Graph_Based_Method
	graph_based_method(dcd, pdb, filename, jump=6, min_number_of_samples=9)

	#mapping the folding landscape
	#get_metastable_states(filename)
	
	#print gbu.evaluate_metastable_states(jump=6)

	#print choose_metastability_criteria(pdb, filename, jump = 6)

if __name__ == '__main__':
  main()
