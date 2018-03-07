import numpy as np
import dill

def convert(file):

	np_arr = []
	with open(file) as f:
		lines = f.readlines()
		for line in lines:
			kvt = (line.strip()).split(",")
			np_arr.append(kvt)

	out_file = open(file+"_npz","wb")
	dill.dump(np.array(np_arr), out_file)

if __name__ == "__main__":
	convert("train.vocab")
	convert("eval.vocab")