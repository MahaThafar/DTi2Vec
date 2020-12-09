'''
*******************************************************
1- Get the dataset type (defualt = 'nr' dataset)
2- Read dataset with all needed information
3- generate all (drug, target) pairs with their labels
*******************************************************
'''
import argparse
import numpy as np
import collections
#-----------------------------------------

def parse_args():

	parser = argparse.ArgumentParser(description="Run DTIs code")
	parser.add_argument('--data', type=str, default='nr',  help='Choose one of the datasets nr,gpcr, ic, e')
	parser.add_argument('--classifier', type=str, default='xgbc',  help='Choose the classifiers: Adaboost (ab), or XGBoost (xgbc))')
	parser.add_argument('--func', type=str, default='Concat',  help='Choose one of the fusion function: Concat, Hadamard, WL1, AVG')

	return parser.parse_args()
#-----------------------------------------

def tree():
    return collections.defaultdict(tree)
#-----------------------------------------

def get_drugs_targets_names(DT):
	# remove the drugs and targets names from the matrix
	DTIs = np.zeros((DT.shape[0]-1,DT.shape[1]-1))

	drugs = []
	targets = []
	for i in range(1,DT.shape[0]):
	    for j in range(1,DT.shape[1]):
	        targets.append(DT[i][0])
	        drugs.append(DT[0][j])
	        DTIs[i-1][j-1] = DT[i][j]

	# to remove duplicate elements       
	targets = sorted(list(set(targets)))
	drugs = sorted(list(set(drugs)))
	DTIs = np.array(DTIs, dtype=np.float64)

	print('Number of drugs:',len(drugs))
	print('Number of targets:', len(targets))

	return drugs, targets, DTIs
#-------------------------------------------------------------------------

def load_datasets(data):
	
	# read the interaction matrix
	DrugTargetF = "Input/"+data+"/"+data+"_admat_dgc.txt"
	DrugTarget = np.genfromtxt(DrugTargetF, delimiter='\t',dtype=str)

	# get all drugs and targets names with order preserving
	all_drugs, all_targets, DTIs = get_drugs_targets_names(DrugTarget)

	## Create R (drug, target, label) with known and unknown interaction
	R = tree()

	# Get all postive drug target interaction R
	with open('Input/'+data+'/R_'+data+'.txt','r') as f:
	    for lines in f:
	        line = lines.split()
	        line[0]= line[0].replace(":","")
	        R[line[1]][line[0]] = 1
	#######################################################################
	#build the BIG R with all possible pairs and assign labels
	label = []
	pairX = []
	for d in all_drugs:
		for t in all_targets:
			p = d, t
            # add negative label to non exit pair in R file
			if R[d][t] != 1:
				R[d][t] = 0
				l = 0
			else:
				l = 1

			label.append(l)
			pairX.append(p)

    # prepare X = all (dr, tr) pairs, Y = labels
	X = np.asarray(pairX)
	Y = np.asarray(label)
	print('dimensions of all pairs', X.shape)

	return all_drugs, all_targets, DTIs, R, X, Y
#----------------------------------------------------------------------------#
##### EOF ####################################################################