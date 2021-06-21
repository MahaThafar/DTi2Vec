# coding: utf-8
# All needed packages
import argparse
import pandas as pd
import math as math
import numpy as np
import csv
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

from xgboost import XGBClassifier
import xgboost as xgb

# Import my files
from load_datasets import *
from training_functions import *
######################################## START MAIN #########################################
#############################################################################################

def main():

# get the parameters from the user
	
	args = parse_args()
	data ='nr'
	func = 'WL1'
	
	print(args.func)
	## get the start time to report the running time
	t1 = time.time()

    ### Load the input data - return all pairs(X) and its labels (Y)..
	allD, allT, DrTr, R, X, Y = load_datasets(args.data)

    # create 2 dictionaries for drugs. the keys are their order numbers
	drugID = dict([(d, i) for i, d in enumerate(allD)])
	targetID = dict([(t, i) for i, t in enumerate(allT)])
    #-----------------------------------------
    ###### Define different classifiers

	xgbc = XGBClassifier(base_score=0.5, booster='gbtree',eval_metric ='error',objective='binary:logistic',
	                    gamma=0,learning_rate = 0.1, max_depth = 7,n_estimators = 600,
	                    tree_method='auto',min_child_weight =4,subsample=0.8, colsample_bytree = 0.9,
	                    scale_pos_weight=1,max_delta_step=1,seed=10, nthread=50)

	#________________________________________________________________
	# 10-folds Cross Validation/ new drug prediction setting

	# all evaluation lists
	correct_classified = []
	ps = []
	recall = []
	roc_auc = []
	average_precision = []
	f1 = []
	Pre = []
	Rec = []
	AUPR_TEST = []
	TN = []
	FP = []
	FN = []
	TP = []

	# Index variables prepared
	all_pair_numbers = len(allD)*len(allT)
	all_matrix_index = []
	index_dictionary_X = {}
	index_counter = 0

	for i in range(len(allD)):
	    for j in range((len(allT))):
	        all_matrix_index.append([i,j])
	        index_dictionary_X[i,j] = index_counter
	        index_counter = index_counter + 1

	# 10-folds Cross Validation for NEW DRUGS
	new_drug_cv_data = new_d_cv_setting(DrTr, 100)

	# fold counter
	foldCounter = 1 
	for fold in new_drug_cv_data:

		#create test_idx and train_idx for X and Y
		train_index = []
		test_index = []
		features = []
		class_labels = []

		print("*** Working with Fold %i :***" %foldCounter)

		#first thing with R train to remove all edges in test (use it when finding path)
		for index in fold[1]:
		    i = index[0]
		    j = index[1]
		    test = index_dictionary_X[i,j]
		    test_index.append(test)

		for idx in range(len(all_matrix_index)):
		    if (idx not in test_index):
		        train_index.append(idx)


		print('train sample length', len(train_index))
		print('test sample length', len(test_index))
		##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		# insert node2vec code here to generate embedding in the same code.....

		# we generated new embeddings for new drug settings, write them into file, read it for new drug evauation

		#------------------------------ node2vec ------------------------------

		# Working with feature vector
		targets ={}
		drugs ={}
		fileName = 'EMBED/newDrug_EMBED/'+args.data+'/EmbeddingFold_'+str(foldCounter)+'.txt'

		## ReadDT feature vectore that came after applying n2v on allGraph including just R_train part
		with open(fileName,'r') as f:
		    #line =f.readline()# to get rid of the sizes
		    for line in f:
		        line = line.split()
		        line[0]= line[0].replace(":","")
		        # take the protien name as key (like dictionary)
		        key = line[0]
		        # remove the protien name to take the remaining 128 features
		        line.pop(0)
		        if key in allT:
		            targets[key] = line
		        else:
		        #key in allD and its feature: 
		            drugs[key] = line
		            
		### Create FV for drugs and for targets
		FV_drugs = []
		FV_targets = []

		for t in allT:
		    FV_targets.append(targets[t])

		for d in allD:
		    FV_drugs.append(drugs[d])  

		# drug node2vec FV, and target node2vec FV
		FV_targets = np.array(FV_targets, dtype = float)
		FV_drugs = np.array(FV_drugs, dtype = float) 

		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		# Build the feature vector FV
		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		# # First operation: Concatenate FVs
		FV_C = []
		concatenateFV = []
		class_labels = []
		DT_pair = []
		for d in allD:
		    for t in allT:
		        dt = d,t
		        DT_pair.append(dt)

		        features = drugs[d] + targets[t]
		        triples = d,t, features
		        FV_C.append(triples)
		        concatenateFV.append(features)
		        # same label as the begining
		        label = R[d][t]
		        class_labels.append(label)

		if(args.func =='Concat'): 
			XX = concatenateFV

	    #>>>>>>>>>>>>>>>>>>>>>>>>>>>
		if(args.func =='Hadmard'): 
		# # Second operation: element wise multiplication (Hadamard Multiplication)
		    HadmardFV = []
		    DrTr_mul = {}
		    counter = 0
		    for d,i in zip (allD, range(len(allD))):
		        for t, j in zip(allT, range(len(allT))):
		            mul = FV_drugs[i,:] * FV_targets[j,:]
		            HadmardFV.append(list((mul)))
		            DrTr_mul[d,t] = mul

		    XX = HadmardFV

	    #>>>>>>>>>>>>>>>>>>>>>>>>>>>
		# #forth operation
		if(args.func =='WL1'): 
		    DrTr_WL1 = []
		    DrTr_dict_WL1 = {}
		    for d,i in zip (allD, range(len(allD))):
		        for t, j in zip(allT, range(len(allT))):
		            abs_sub = abs(FV_drugs[i,:] - FV_targets[j,:])
		            DrTr_WL1.append(list((abs_sub)))
		            DrTr_dict_WL1[d,t] = abs_sub
		  
		    XX = DrTr_WL1

		#>>>>>>>>>>>>>>>>>>>>>>>>>>>
		# #fifth operation
		if(args.func =='WL2'): 
			DrTr_WL2 = []
			DrTr_dict_WL2 = {}
			for d,i in zip (allD, range(len(allD))):
				for t, j in zip(allT, range(len(allT))):
					abs_sub = abs(FV_drugs[i,:] - FV_targets[j,:])
					abs_sub = abs_sub**2
					DrTr_WL2.append(list((abs_sub)))
					DrTr_dict_WL2[d,t] = abs_sub

			XX = DrTr_WL2
		#>>>>>>>>>>>>>>>>>>>>>>>>>>>


		## Start Classification Task
		# featureVector and labels for each pair
		XX = np.asarray(XX)
		YY = np.array(Y)

		#Apply normalization using MaxAbsolute normlization
		max_abs_scaler = MinMaxScaler()
		X_train = max_abs_scaler.fit(XX[train_index]) 
		X_train_transform = max_abs_scaler.transform(XX[train_index])

		X_test_transform = max_abs_scaler.transform(XX[test_index])

		# Apply sampling techniques for the trainnig data
		ros = RandomOverSampler(random_state= 10)
		X_res, y_res= ros.fit_sample(X_train_transform, YY[train_index])


		# fit the model and predict
		xgbc.fit(X_res, y_res)
		predictedClass = xgbc.predict(X_test_transform)
		predictedScore = xgbc.predict_proba(X_test_transform)[:, 1]


		# ------------------- Print Evaluation metrics for each fold --------------------------------
		print("@@ Validation and evaluation of fold %i @@" %foldCounter)
		print(YY[test_index].shape, predictedClass.shape)

		cm = confusion_matrix(YY[test_index], predictedClass)
		TN.append(cm[0][0])
		FP.append(cm[0][1])
		FN.append(cm[1][0])
		TP.append(cm[1][1])
		print("Confusion Matrix for this fold")
		print(cm)

		print("Correctly Classified Instances: %d" %accuracy_score(Y[test_index], predictedClass, normalize=False))
		correct_classified.append(accuracy_score(Y[test_index], predictedClass, normalize=False))

		#print("Precision Score: %f" %precision_score(Y[test_index], predictedClass))
		ps.append(precision_score(Y[test_index], predictedClass,average='weighted'))

		#print("Recall Score: %f" %recall_score(Y[test_index], predictedClass)
		recall.append(recall_score(Y[test_index], predictedClass, average='weighted'))

		print("F1 Score =  %f" %f1_score(Y[test_index], predictedClass, average='weighted'))
		f1.append(f1_score(Y[test_index], predictedClass,average='weighted'))

		print("AUC =  %f" %roc_auc_score(Y[test_index], predictedScore))
		roc_auc.append(roc_auc_score(Y[test_index], predictedScore))

		p, r, _ = precision_recall_curve(Y[test_index],predictedScore,pos_label=1)
		aupr = auc(r, p)
		print("AUPR  = %f" %aupr)
		AUPR_TEST.append(aupr)

		Pre.append(p.mean())
		Rec.append(r.mean())
		average_precision.append(average_precision_score(Y[test_index], predictedScore))

		print(classification_report(Y[test_index], predictedClass))
		print('------------------------------------------------------')
		foldCounter += 1
		#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

	#--------------------------------------------------------------------
	############# Evaluation Metrics ####################################
	# Confusion matrix for all folds
	ConfMx = np.zeros((cm.shape[0],cm.shape[0]))
	ConfMx[0][0] = str( np.array(TN).sum() )
	ConfMx[0][1] = str( np.array(FP).sum() )
	ConfMx[1][0] = str( np.array(FN).sum() )
	ConfMx[1][1] = str( np.array(TP).sum() )

	### Print Evaluation Metrics.......................
	print("Result(Correct_classified): " + str( np.array(correct_classified).sum() ))
	print("Results:precision_score = " + str( np.array(ps).mean().round(decimals=3) ))
	print("Results:recall_score = " + str( np.array(recall).mean().round(decimals=3) ))
	print("Results:f1 = " + str( np.array(f1).mean().round(decimals=3) ))
	print("Results:roc_auc = " + str( np.array(roc_auc).mean().round(decimals=3) ))
	print("Results: AUPR on Testing auc(r,p) = " + str( np.array(AUPR_TEST).mean().round(decimals=3)))
	print("Results: Std of 10 folds aupr on Testing = " + str( np.std(AUPR_TEST)))
	print("Confusion matrix for all folds")
	print(ConfMx)
	print('_____________________________________________________________')
	print('Running Time for the whole code:', time.time()-t1)  
	print('_____________________________________________________________')  
	######################################################################################

#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    main()
#####-------------------------------------------------------------------------------------------------------------
####################### END OF THE CODE ##########################################################################