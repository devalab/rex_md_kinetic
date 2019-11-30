import shelve,re

class Autovivification(dict):
	def __getitem__(self, item, **args):
		try:
			return dict.__getitem__(self, item)
		except KeyError:
			value = self[item] = type(self)()
           		return value


class disjoint():
    def __init__(self, elements):
        self.parent = {}
        for i in elements:
            self.parent[i] = i
        self.elements = elements

    def union(self, i, j):
        ipar = self.findparent(i)
        jpar = self.findparent(j)
        self.parent[ipar] = jpar

    def findparent(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.findparent(self.parent[i])
        return self.parent[i]

    def compress(self):
        for i in self.elements:
            self.findparent(i)

    def print_details(self):
        for i in set(self.parent):
            print i, ": ",
            for j in self.elements:
                if (self.findparent(j) == i):
                    print j,
            print

    def get_components(self, parent):
        temp = []
        for j in self.elements:
            if (self.findparent(j) == parent):
                temp.append(j)
        return temp

    def save_structure(self):
        """
        dynnamic clusters is a file that strores a dict mapping of each dynamic cluster id and number of frames which belong to that cluster.
        """
        d = shelve.open("dynamic_clusters")
        d.clear()
        for i in set(self.parent):
            temp = []
            for j in self.elements:
                if (self.findparent(j) == i):
                    temp.append(j)
            if (len(temp) > 0):
                d[str(i)] = temp
        d.close()

    def get_clusters(self, cluster_probability):
        d = shelve.open("dynamic_clusters")
        parent = {}
        for i in d:
            maxindex = d[i][0]
            currentmax = 0.0
            for index in d[i]:
                if (currentmax < cluster_probability[index]):
                    maxindex = index
                    currentmax = cluster_probability[index]

            for index in d[i]:
                parent[index] = maxindex

        return parent


class psf_structure():
	def __init__(self):
		self.atoms = []
		
	def create(self,psf_filename):
		f = open(psf_filename, "r").readlines()
		atom_start = 0
		for line in f:
		    if re.search("!NATOM", line):
		     	atom_start = 1
		    if re.search("!NBOND", line):
			break #for now only non-bonded interactions are considered
		    if atom_start == 1:
			cols = line.split()
			try:
			    self.atoms.append(atom(int(cols[0].strip()),cols[1].strip(),int(cols[2].strip()),cols[3].strip(),cols[4].strip(),cols[5].strip(),float(cols[6].strip()),float(cols[7].strip())))
			except:
			    continue

	def get_atom(self,index):
		return self.atoms[index]

class atom():
	def __init__(self, a_id, segment_name, resid, resname, atom_name, atom_type, charge, mass):
		self.id = int(a_id)
		self.segname = segment_name
		self.resid = int(resid)
		self.resname = resname
		self.name = atom_name
		self.type = atom_type
		self.charge = float(charge)
		self.mass = float(mass)

	def show_details(self):
		print "Id: ", self.id
		print "Segname: ", self.segname
		print "Resid: ",self.resid
		print "Resname: ",self.resname
		print "Name: ",self.name 
		print "Type: ",self.type 
		print "Charge: ",self.charge
		print "Mass: ",self.mass 


class parameters():
	def __init__(self):
		self.vdw_parameters = {}

	def populate_vdw(self, filename):
		vdw_on = False
		f = open(filename, "r").readlines()
		for lines in f:
			if re.search("NONBONDED", lines):
				vdw_on = True

			if re.search("HBOND",lines):
				vdw_on = False

			if vdw_on == True:
				cols = lines.split()
				try:
				    self.add_vdw_param(cols[0].strip(), cols[2].strip(), cols[3].strip())
				except:
				    continue
		

	def add_vdw_param(self,atom_type,epsilon,rmin):
		self.vdw_parameters[atom_type] = vdw(atom_type, epsilon, rmin)
		
		return

class vdw():
	def __init__(self, atom_type, epsilon, rmin):
		self.type = atom_type
		self.epsilon = float(epsilon)
		self.rmin = float(rmin) #charmm par files has rmin/2


	def show_details(self):
		print "Type: ",self.type 
		print "Epsilon: ",self.epsilon
		print "Rmin: ",self.rmin 







		
	
