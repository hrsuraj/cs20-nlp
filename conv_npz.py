import numpy as np
import dill

def convert(file):

	wdict = {}
	with open(file) as f:
		lines = f.readlines()
		for line in lines:
			kvt = (line.strip()).split(" ")
			vlist = kvt[1].split(",")
			val_list = [float(i) for i in vlist]
			wdict[kvt[0]] = np.array(val_list)

	out_file = open(file+"_dict","wb")
	dill.dump(wdict, out_file)

if __name__ == "__main__":
	convert("english.vec")
	convert("spanish.vec")


