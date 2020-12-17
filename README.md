# DTi2Vec: Drug-Target interaction prediction using network embedding and esemble learning


Submitted: 16 December 2020                                      



----
This code is implemented using Python 3.7

For any qutions please contact the first author:


  Maha Thafar

Email: maha.thafar@kaust.edu.sa

Computer, Electrical and Mathematical Sciences and Engineering Division (CEMSE), Computational Bioscience Research Center, Computer (CBRC), King Abdullah University of Science and Technology (KAUST).

----

## Getting Started

### Prerequisites:

There are several required Python packages to run the code:
- gensim (for node2vec code)
- numpy
- Scikit-learn
- imblearn
- pandas
- xgboost

These packages can be installed using pip or conda as the follwoing example
```
pip install -r requirements.txt
```
----

### Files Description:
#### *There are Three folders:*

  **1.(Input) folder:** 
  that includes four folder for 5 datasets include: 
   - Nuclear Receptor dataset (nr),
   - G-protein-coupled receptor (gpcr),
   - Ion Channel (ic), 
   - Enzyme (e)
   - FDA_DrugBank (DrugBank)
     which each one of them has all required data of drug-target interactions (in Adjacency matrix and edgelist format) and drug-drug similarity and target-target similarity in (square matrix format)
  
  **2.(EMBED) folder:**
  that has also five folders coressponding for five datasets,
     each folder contains the generated node2vec Embedding file for each fold of training data (coressponding to the same seed of CV in the main node2vec code)
     
  **3.(Novel_DTIs) folder:**
  that has also four folders coressponding for four datasets, 
     to write the novel DTIs (you should create directory for each dataset)
  
---
#### *There are 2 files of the implementation:*
(Four main functions, one main for each dataset, and the other functions are same for all datasets which are imported in each main function)

- **load_datasets.py** --> to read the input data for each dataset sperately


- **main function**
> - DTIs_Main.py


---
## Installing:

To get the development environment runining, the code get one parameter from the user which is the dataset name (the defual dataset is nr)
run:



------------------
### For citation:
---


