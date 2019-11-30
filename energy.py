import mdtraj as md
import theano, time
import numpy as np
import theano.tensor as T
import auxillary_data_structures as aux
import misc

def apply_pbc_conditions(sel1, sel2, pbcsize):
	m = sel1.shape
	n = sel2.shape


	X = T.dmatrix("X")
	Y = T.dmatrix("Y")
	
	pbc_s = T.dvector("pbc")
	displacement_vector = X.dimshuffle(0,'x',1) - Y.dimshuffle('x',0,1)

	displacement_vector_new = T.switch(T.gt(displacement_vector, pbc_s * 0.5), displacement_vector - pbc_s, displacement_vector)
	minimum_image_vector = T.switch(T.lt(displacement_vector_new, pbc_s * -0.5), displacement_vector_new + pbc_s, displacement_vector_new)	
	
	
	pbc_func = theano.function([X, Y, pbc_s], minimum_image_vector)
	
	displacement_matrix = pbc_func(sel1, sel2, pbcsize)

	return displacement_matrix

def create_distance_matrix(displacement_matrix):
	X = T.dtensor3("d")
	dist_matrix = T.sqrt(T.sum(X*X, axis=-1))


	dist_func = theano.function([X], dist_matrix)
	
	distance_matrix = dist_func(displacement_matrix)

	return distance_matrix

def get_atom_indices_within_cutoff(sel1, sel2, pbc_size, cutoff):
	#old_time = time.time()
	displacement_matrix = apply_pbc_conditions(sel1, sel2, pbc_size)
	distance_matrix = create_distance_matrix(displacement_matrix)

	indices = np.where( distance_matrix < cutoff )
	#print "Time Elapsed:", time.time() - old_time
	
	return distance_matrix, indices


def calc_elec_energy(sel1, sel2, psf_struct, distance_matrix, ron, roff):
	"""
	E = (qi * qj)/rij
	"""
	COULOMB = 332.0636 #taken from VMD source code
	charge_sel1 = []
	for atom_index in sel1:
		charge_sel1.append(psf_struct.get_atom(atom_index).charge)

	
	charge_sel2 = []
	for atom_index in sel2:
		charge_sel2.append(psf_struct.get_atom(atom_index).charge)

	charge_sel1 = np.array(charge_sel1)
	charge_sel2 = np.array(charge_sel2)
	
	Q1 = T.dvector("Q1")
	Q2 = T.dvector("Q2")
	R = T.dvector("R")
	
	efac = 1.0-(R*R/(roff*roff));


	E = (COULOMB*Q1*Q2*efac*efac)/R
	E_nn = T.sum(T.switch(T.gt(R, roff), 0.0, E))

	
	ener = theano.function([Q1, Q2, R], E_nn)
	return ener(charge_sel1,charge_sel2,distance_matrix)

def calc_vdw_energy(sel1, sel2, psf_struct, distance_matrix, ron, roff, par_struct):
	"""
	E = ((RMIN/R)**12 - 2*((RMIN/R)**6))*EPS
	"""

	type_sel1 = []
	for atom_index in sel1:
		type_sel1.append(psf_struct.get_atom(atom_index).type)

	
	type_sel2 = []
	for atom_index in sel2:
		type_sel2.append(psf_struct.get_atom(atom_index).type)


	eps_sel1 = []
	rmin_sel1 = []
	eps_sel2 = []
	rmin_sel2 = []
	for index in xrange(len(type_sel1)):
		eps_sel1.append(par_struct.vdw_parameters[type_sel1[index]].epsilon)	
		rmin_sel1.append(par_struct.vdw_parameters[type_sel1[index]].rmin)	

		eps_sel2.append(par_struct.vdw_parameters[type_sel2[index]].epsilon)	
		rmin_sel2.append(par_struct.vdw_parameters[type_sel2[index]].rmin)	


	eps_sel1 = np.array(eps_sel1)
	rmin_sel1 = np.array(rmin_sel1)
	eps_sel2 = np.array(eps_sel2)
	rmin_sel2 = np.array(rmin_sel2)

	#print eps_sel1, rmin_sel1, eps_sel2, rmin_sel2

	x = T.dvector("x")
	y = T.dvector("y")
	
	EPS = T.sqrt(x*y)

	a = T.dvector("a")
	b = T.dvector("b")
	RMIN = a + b
	
	R = T.dvector("R")
	
	E = ((RMIN/R)**12 - 2*((RMIN/R)**6))*EPS
	E = T.switch(T.gt(R, ron), (E*((roff - R)*(roff - R)*(roff + 2*R -3*ron)))/((roff - ron)*(roff - ron)*(roff - ron)), E)
	E = T.sum(T.switch(T.gt(R, roff), 0.0, E))

	ener = theano.function([x, y, a, b, R], E)
	return ener(eps_sel1, eps_sel2, rmin_sel1, rmin_sel2, distance_matrix)

def get_params(filename):
	par_struct = aux.parameters()
	temp = open(filename, "r").readlines()
	for line in temp:
		par_struct.populate_vdw(line.strip())
	return par_struct

def calc_interaction_energy(dcd_filename, psf_filename, prm_file, pbc_size_frames, first_sel, second_sel, cutoff = 12.0, ron=10.0, roff=12.0, energy_output_file = "energy_output.dat"):
	psf_struct = aux.psf_structure()
	psf_struct.create(psf_filename)

	par_struct = get_params(prm_file)

	print "size of first selection:", first_sel.shape, 
	print "size of second selection:", second_sel.shape
	traj = md.formats.DCDTrajectoryFile(dcd_filename)
	output_file = open(energy_output_file, "w")
	#uncomment below line to check whether selections working perfectly!
	#print first_sel, second_sel

	old_time = time.time()
	index = 0
	while(1):
		if index%100 == 0:
			print index
			print "Time Elapsed:", time.time() - old_time
			old_time = time.time()
		try:
			model_atom_positions = traj.read(n_frames=1)[0][0]
		except:
			break

		sel1 = model_atom_positions[first_sel]
		sel2 = model_atom_positions[second_sel]

		if pbc_size_frames.shape[0] > 1:
			pbc_size = np.array(pbc_size_frames[index])
		else:
			pbc_size = np.array(pbc_size_frames[0])
		
		distance_matrix, temp = get_atom_indices_within_cutoff(sel1, sel2, pbc_size, cutoff)

		dist_mat_within_cutoff = distance_matrix[temp]

		set1 = first_sel[temp[0]]
		set2 = second_sel[temp[1]]
		
		elec_energy = calc_elec_energy(set1, set2, psf_struct, dist_mat_within_cutoff, ron, roff)
		

		vdw_energy = calc_vdw_energy(set1, set2, psf_struct, dist_mat_within_cutoff, ron, roff, par_struct)
		output_file.write(str(index) + " " + "%.4f" % elec_energy + " " + "%.4f" % vdw_energy + " " + "%4f" % (vdw_energy+elec_energy) + "\n")
		
		index +=1
	return
