import numpy as np
import mdtraj as md
import joblib, math, sys, re, heapq, energy
import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import auxillary_data_structures as aux

def exp_func(x, b):
	return np.exp(-b * x)

def find_indices_of_clusters(labels, index):
	temp = []
	for i in range(labels.shape[0]):
		if int(labels[i]) == index:
			temp.append(int(i))
	return temp


def get_average_structure(frame_indices, frames):
    temp = np.mean(frames[np.array(frame_indices)], axis=0)
    return temp


def get_dist(a, b):
    c = a - b
    for i in range(a.shape[0]):
        dot_product = np.array(np.dot(c[i], c[i].T))
        try:
            d = np.hstack((d, dot_product))
        except:
            d = dot_product
    return np.sum(d)


def get_closest_structure_index(frame_indices, frames, avg_structure):
    dist = float("inf")
    index = 0
    counter = 0
    # print frame_indices
    for i in frames[np.array(frame_indices)]:
        temp_dist = get_dist(i, avg_structure)
        if temp_dist <= dist:
            dist = temp_dist
            index = counter

        counter += 1
    return frame_indices[index]


def write_pdb_file(temp, pdb, cluster_id,type_of_cluster):
    temp *= 10.0
    pdb_file = open(pdb, "r").readlines()
    temp_file = open(type_of_cluster + "_" + str(cluster_id) + ".pdb", "w")
    for line in pdb_file:
        if re.search("CRYST1|END", line):
            temp_file.write(line)
        else:
            cols = line.split()
            temp_line = line[:30]
            xpos = str(np.round(temp[int(cols[1]) - 1][0], 3))
            num_of_zeros_needed_x = len(xpos.split('.')[1])  # putting trailing zeroes after decimal point
            while num_of_zeros_needed_x < 3:
                xpos += "0"
                num_of_zeros_needed_x += 1

            for space in range(8 - len(xpos)):
                temp_line += " "
            temp_line += str(xpos)

            ypos = str(np.round(temp[int(cols[1]) - 1][1], 3))

            num_of_zeros_needed_y = len(ypos.split('.')[1])
            while num_of_zeros_needed_y < 3:
                ypos += "0"
                num_of_zeros_needed_y += 1

            for space in range(8 - len(ypos)):
                temp_line += " "

            temp_line += str(ypos)

            zpos = str(np.round(temp[int(cols[1]) - 1][2], 3))
            num_of_zeros_needed_z = len(zpos.split('.')[1])
            while num_of_zeros_needed_z < 3:
                zpos += "0"
                num_of_zeros_needed_z += 1

            for space in range(8 - len(zpos)):
                temp_line += " "
            temp_line += str(zpos)

            temp_line += line[54:]
            line = temp_line
            temp_file.write(line)
    temp_file.close()

def distance(a,b):
    a = np.array(a)
    b = np.array(b)
    dist = 0.0
    for i in range(a.shape[0]):
        dist += (a[i] - b[i])*(a[i] - b[i])
    return math.sqrt(dist)


def most_probable_structure_in_cluster(frame_indices, frames, pdb, cluster_id, type_of_cluster, dcd_pkl_filename, jump=1):
    # this function finds the most probable 2-D location of each cluster
    array = frames[frame_indices]
    if array.shape[0] <= 3:
        mean_point = np.mean(array, axis=0)
    else:
        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': np.logspace(-1, 0, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(array)

        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
        sampling_points = kde.sample(n_samples=10000, random_state=20)
	#Z = kde.score_samples(sampling_points)
        #Z = Z.reshape(X.shape)

        #np.save("prob_density"+base, Z)
        #index = np.unravel_index(Z.argmax(), Z.shape)
        #print Z[index], index
        #print "most probable values:", X[index], Y[index]

        mean_point = np.mean(sampling_points, axis=0)
	#mean_point = sampling_points[index]	

    closest_structure_index = get_closest_structure_index(frame_indices, frames, mean_point)

    for x in frame_indices:
   	 print "mol addfile " + dcd_pkl_filename[:3] + ".dcd first " + str(x*jump) + " last " + str(x*jump) +" waitfor all" 
    print
    dcd_array = preprocessing.load_residues(dcd_pkl_filename)[::jump]
    temp = dcd_array[closest_structure_index]
   # print "mol addfile " + dcd_pkl_filename[:3] + ".dcd first " + str(closest_structure_index*jump) + " last " + str(closest_structure_index*jump) +" waitfor all" 
    write_pdb_file(temp,pdb,cluster_id,type_of_cluster)
    return mean_point

def djkstra_widest_path(graph,start,end):
    dist = [-1.0*float("inf")] * graph.shape[0]
    pre = [-1] * graph.shape[0]
    vis = [0] * graph.shape[0]
    number_of_unvisited_vertices = graph.shape[0]
    dist[start] = float("inf")
    unvisited_vertices = []
    k = 1.0*float("inf")

    for i in range(graph.shape[0]):
        if i != start:
            item = [k, i, start]
        else:
            item = [-1.0*k, i, i]
        heapq.heappush(unvisited_vertices, item)
    source = -1
    while (number_of_unvisited_vertices > 0):
        vertex = heapq.heappop(unvisited_vertices)
        distance = -1.0*vertex[0]
        ver = vertex[1]
	 	
        if vis[ver] != 0:
            continue
	
        number_of_unvisited_vertices -= 1
        vis[ver] = 1
        pre[ver] = vertex[2]
        dist[ver] = distance
        source = ver
        for i in range(graph[ver].shape[0]):
            if i != ver:
                new_dist = max(min(dist[ver],graph[ver][i]), dist[i])
                if new_dist > dist[i]:
                    item = [-1.0*new_dist, i, ver]
                    heapq.heappush(unvisited_vertices, item)

    endvertex = end
    shortest_path = []
    shortest_path.append(end)
    while (True):
        prevvertex = pre[endvertex]
        shortest_path.append(prevvertex)
        if prevvertex == start:
            break
        endvertex = prevvertex

    # print "dist[end]", dist[end]
    return shortest_path[::-1]


def djkstra(graph, start, end):
    dist = [float("inf")] * graph.shape[0]
    pre = [-1] * graph.shape[0]
    vis = [0] * graph.shape[0]
    number_of_unvisited_vertices = graph.shape[0]
    dist[start] = 0.0
    unvisited_vertices = []
    k = float("inf")

    for i in range(graph.shape[0]):
        if i != start:
            item = [k, i, start]
        else:
            item = [0, i, i]
        heapq.heappush(unvisited_vertices, item)
    source = -1
    while (number_of_unvisited_vertices > 0):
        vertex = heapq.heappop(unvisited_vertices)
        distance = vertex[0]
        ver = vertex[1]

        if vis[ver] != 0:
            continue

        number_of_unvisited_vertices -= 1
        vis[ver] = 1
        pre[ver] = vertex[2]
        dist[ver] = distance
        source = ver
        for i in range(graph[ver].shape[0]):
            if i != ver:
                new_dist = min(dist[ver] + graph[ver][i], dist[i])
                if new_dist < dist[i]:
                    item = [new_dist, i, ver]
                    heapq.heappush(unvisited_vertices, item)

    endvertex = end
    shortest_path = []
    shortest_path.append(end)
    while (True):
        prevvertex = pre[endvertex]
        shortest_path.append(prevvertex)
        if prevvertex == start:
            break
        endvertex = prevvertex

    # print "dist[end]", dist[end]
    return shortest_path[::-1]


def get_cluster_ids_for_start_and_end(initial_id, cluster_indices, type_of_clustering):
    start = initial_id  # initial frame is folded structure
    # choose structure with maximum SASA as unfolded structure
    unfolded_structure_cluster_id = start
    max_sasa = 0.0
    for index in cluster_indices:
        try:
            f = md.load_pdb(type_of_clustering + "_" + str(index) + ".pdb")
        except:
            continue
        sasa = md.shrake_rupley(f)
        total_sasa = sasa.sum(axis=1)
        if total_sasa > max_sasa:
            max_sasa = total_sasa
            unfolded_structure_cluster_id = index

    end = unfolded_structure_cluster_id
    #       end =  gaussian_cluster_ids[str(frames.shape[0] - 1)]
    return start, end


def write_dcd(frames, path, type_of_cluster):
    f = md.formats.DCDTrajectoryFile(type_of_cluster + "_unfolded_traj.dcd", "w")
    positions = [frames[0] * 10.0]
    # print positions.shape
    for p in path:
        g = md.load_pdb(type_of_cluster + "_" + str(p) + ".pdb")
        positions.append(g.xyz[0] * 10.0)
    positions = np.array(positions)

    f.write(positions)
    return f

def write_pdb(frames,path, type_of_cluster):
    positions = [frames[0] * 10.0]
    index = 0
    # print positions.shape
    for p in path:
	f = md.formats.PDBTrajectoryFile(type_of_cluster + "_path_" + str(index) + ".pdb", "w")
        g = md.load_pdb(type_of_cluster + "_" + str(p) + ".pdb")
        positions = (g.xyz[0] * 10.0)
	index += 1
    	f.write(positions,g.topology)
    return f

def convert_to_list(modified_graph):
    """
    This funtion returns a list representation of an adjacency matrix from its double dict (Autovivification) form
    :param modified_graph:
    :return:
    """
    adjacency_list = []
    for i in sorted(modified_graph.keys()):
        temp = []
        for j in sorted(modified_graph.keys()):
            temp.append(modified_graph[i][j])
        adjacency_list.append(temp)
    return np.array(adjacency_list)

def split_trajectory(dcd, number_of_frames_in_each_split):
	i = 0
	frames = []
	dcd = md.formats.DCDTrajectoryFile(dcd)
	while(1):
		try:
			old_time = time.time()
			traj = dcd.read(n_frames=number_of_frames)
			new_time = time.time()			
			print "Time taken for reading 1500 frames:", new_time - old_time
		except:
			break
		
		model_atom_positions = traj[0]
		
		dcd_file = md.formats.DCDTrajectoryFile("temp_"+str(i)+".dcd", "w")
		positions = np.array(model_atom_positions)
		dcd_file.write(positions)
	
		new_time = time.time()			
		print "Time for writing taken for 1500 frames:", new_time - old_time
		print dcd.tell()
		i+=1	
	return


def rna_residue_selections(pdb_filename):
	""" this function creates a mapping of atom index and residue it belongs to:"""
	pdb_file = open(pdb_filename,"r").readlines()
	residue_indices = {}
        mol_atom_dict={}
	backbone_indices=[]
	for line in pdb_file: #leaving the first line out as it contains crystal information
		if re.search("ATOM", line):
			cols = line.split()
			if len(cols) > 7:
			  atom_type=cols[2]
			  atom_index=int(cols[1]) - 1	
			  p_atom= "P" in  atom_type
			  if p_atom:
			    backbone_indices.append(atom_index )
			    continue
			  terminal_atom="T" in atom_type
			  if terminal_atom:
			    backbone_indices.append(atom_index)
			    continue
			  non_p_backbone="'" in atom_type
		          if non_p_backbone:
			    backbone_indices.append(atom_index )
			    continue
		          else: 			 
				mol_idx=int(cols[4]) - 1 
				residue_indices[atom_index] =mol_idx
		                if mol_idx in mol_atom_dict:
		                     mol_atom_dict[mol_idx].append(atom_index)
				else:
		                     mol_atom_dict[mol_idx]=[atom_index]     
		          

	return residue_indices , mol_atom_dict,backbone_indices

def get_pbc_dimensions(pbc_size_filename):
        pbc_size = []
        f = open(pbc_size_filename, "r").readlines()
        for line in f[2:]:
                cols = line.split()
                x = float(cols[1].strip())
                y = float(cols[5].strip())
                z = float(cols[9].strip())

                pbc_size.append([x,y,z])
        return pbc_size


def select_pairs_for_nbenergy(dcd_filename, pdb_filename, psf_filename, prm_file, pbc_size_filename, sel1_statement, sel2_statement, cutoff = 12.0, ron=10.0, roff=12.0, energy_output_file = "energy_output.dat"):
	pdb = md.load(pdb_filename)
	topology = pdb.topology
	
	try:
		pbc_size = get_pbc_dimensions(pbc_size_filename)
	except:  #if only one float value is specified for PBC instead of a .xsc file
		pbc_size = np.array([float(pbc_size_filename),float(pbc_size_filename),float(pbc_size_filename)])
		pbc_size = pbc_size.reshape((1,-1))
	
	residue_indices_dict , mol_atom_dict, backbone_indices = rna_residue_selections(pdb_filename)

	#table, bonds = topology.to_dataframe()
	#print(table.ix[21575])

	if re.search("RNA backbone", sel1_statement):
		first_sel = np.array(backbone_indices)
	else:
		first_sel = topology.select(sel1_statement)
		first_sel = first_sel[np.invert(np.in1d(first_sel, np.array(backbone_indices)))]
	
	second_sel = topology.select(sel2_statement)	

	energy.calc_interaction_energy(dcd_filename, psf_filename, prm_file, pbc_size, first_sel, second_sel, cutoff, ron, roff, energy_output_file)

