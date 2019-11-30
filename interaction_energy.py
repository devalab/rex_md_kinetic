import misc
import sys

#this script is used to calculate non bonded energy terms, viz. electrostatic and lennard jones potential
def main():
	dcd = sys.argv[1] #the dcd file to analyze
	pdb = sys.argv[2] #the pdb file of the initial structure
	psf = sys.argv[3] #the psf file of the initial structure in NAMD X-PLOR format
	prm_file = sys.argv[4] #file contanining path of each parameter file (CHARMM format), each path must be "\n" separated 
	pbc_size = sys.argv[5] # a int or NAMD .xst file path

	#refer to "http://mdtraj.org/latest/atom_selection.html", in case you want to select the RNA backbone molecules in sel1, use "RNA_backbone"
	sel1_statement = sys.argv[6] #first group of atoms
	sel2_statement = sys.argv[7] #second group of atoms, interactions are calculated for each atom in sel1 with each in sel2
	cut = sys.argv[8] #cutoff criteria for the nonbonded interactions, use 12A if not sure 
	output_file = sys.argv[9] #path for output file to be saved
	misc.select_pairs_for_nbenergy(dcd, pdb, psf, prm_file, pbc_size, sel1_statement, sel2_statement, cutoff = float(cut),energy_output_file=output_file)																															



if __name__ == '__main__':
  main()
