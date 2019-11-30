import mdtraj as md
from sklearn.decomposition import PCA
import joblib
import numpy as np

def read_trajectory(dcd, pdb):
    traj = md.load(dcd, top=pdb)
    frames = []
    for i in range(len(traj)):
        model_atom_positions = traj[i].xyz[0]
        frames.append(model_atom_positions)

    return np.array(frames)


def write_residues(frames, filename):
    f = open(filename, "w")
    np.save(f, frames)
    return


def load_residues(filename):
    f = open(filename, "r")
    deserialized_a = np.load(f)
    return deserialized_a


def reduce_dimensions(frames, fit = True):
	frames = frames.reshape((frames.shape[0],-1,))
	print frames.shape
	if fit == True:
		pca = PCA(n_components=0.99)
		pca.fit(frames)
		joblib.dump(pca, 'pca.pkl')
	else:
		pca = joblib.load('pca.pkl')
		reduced_frames = pca.transform(frames)
		print reduced_frames.shape
		write_residues(reduced_frames, 'reduced_dimensions.pkl')


def create_energy_matrix(get_from_file=1, residues=12):
    temp = []
    if get_from_file==1:
        for i in range(1 , residues+1):
            for j in range( i + 1, residues + 1):
                temp = []
                file_name = "inter_"+str(i)+"_"+str(j)+".dat"
                f = open(file_name , "r") . readlines()

                for lines in f:
                    cols = lines.split()
                    # print cols
                    temp.append(float(cols[1].strip()))
                try:
		   
                    energy_matrix = np.vstack((energy_matrix, np.array(temp)))
                except:
                    energy_matrix = np.array(temp)

        # adding the backbone energy to the energy matrix
        file_name = "inter_backbone.dat"
        f = open(file_name, "r").readlines()
        temp = []
        for lines in f:
            cols = lines.split()
            temp.append(float(cols[1]))
        energy_matrix = np.vstack((energy_matrix, np.array(temp)))
        #energy_matrix = energy_matrix.T
	
	for i in range(0 , residues):
		temp = []
		file_name = "inter_"+str(i)+"_water.dat" 
		f = open(file_name , "r") . readlines()

		for lines in f:
		    cols = lines.split()
		    # print cols
		    temp.append(float(cols[3].strip()))
		try:
		    energy_matrix = np.vstack((energy_matrix, np.array(temp)))
		except:
		    energy_matrix = np.array(temp)

	for i in range(0 , residues):
		temp = []
		file_name = "inter_"+str(i)+"_urea.dat" 
		f = open(file_name , "r") . readlines()

		for lines in f:
		    cols = lines.split()
		    # print cols
		    temp.append(float(cols[3].strip()))
		try:
		    energy_matrix = np.vstack((energy_matrix, np.array(temp)))
		except:
		    energy_matrix = np.array(temp)
	
	# adding the backbone energy to the energy matrix
        file_name = "inter_backbone_urea.dat"
        f = open(file_name, "r").readlines()
        temp = []
        for lines in f:
            cols = lines.split()
            temp.append(float(cols[3]))
        energy_matrix = np.vstack((energy_matrix, np.array(temp)))
        #energy_matrix = energy_matrix.T

	# adding the backbone energy to the energy matrix
        file_name = "inter_backbone_water.dat"
        f = open(file_name, "r").readlines()
        temp = []
        for lines in f:
            cols = lines.split()
            temp.append(float(cols[3]))
        energy_matrix = np.vstack((energy_matrix, np.array(temp)))
        energy_matrix = energy_matrix.T

        temp = energy_matrix
        f = open("original_matrix", "w")
        np.save(f,temp)

    elif get_from_file==2:
        f = open("original_matrix", "r")
        temp = np.load(f)

    return np.array(temp)


def normalize_energy_matrix(frames=[], get_from_file=True):
    if get_from_file == False:
        temp = frames
        max_interaction = np.amax(temp[:,:66])
        min_interaction = np.amin(temp[:,:66])
	#print max_interaction, min_interaction
        temp[:, :66] = (2 * temp[:, :66] - (min_interaction + max_interaction)) / (max_interaction - min_interaction)

        max_energy = np.amax(temp[:,66])
        min_energy = np.amin(temp[:,66])										
       # print max_energy, min_energy
	
        temp[:,66] = (2 * temp[:,66] - (min_energy + max_energy)) / (max_energy - min_energy)

	max_interaction = np.amax(temp[:,67:-2])
        min_interaction = np.amin(temp[:,67:-2])
	#print max_interaction, min_interaction
        temp[:,67:-2] = (2 * temp[:,67:-2] - (min_interaction + max_interaction)) / (max_interaction - min_interaction)

	max_energy = np.amax(temp[:,-2:])
        min_energy = np.amin(temp[:,-2:])										
        #print max_energy, min_energy
	
        temp[:,-2:] = (2 * temp[:,-2:] - (min_energy + max_energy)) / (max_energy - min_energy)

        f = open("normalized_original_matrix", "w")
        np.save(f, np.array(temp))
    else:
        f = open("normalized_original_matrix", "r")
        temp = np.load(f)
    return temp

