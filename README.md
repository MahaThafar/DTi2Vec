# DTi2Vec: Drug-Target interaction prediction using network embedding and esemble learning


Submitted: 16 December 2020                                      



----
This code is implemented using Python 3.7

For any qutions please contact the first author:


  Maha Thafar

Email: maha.thafar@kaust.edu.sa

Computer, Electrical and Mathematical Sciences and Engineering Division (CEMSE), Computational Bioscience Research Center, Computer (CBRC), King Abdullah University of Science and Technology (KAUST) - Collage of Computers and Information Technology, Taif University (TU)

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
     - to access the main code of node2vec: https://github.com/aditya-grover/node2vec, or you can install node2vec library using 
     ```
     pip install node2vec
     ```
  
---
#### *There are 2 files of the implementation:*


- **load_datasets.py** --> to read the input data for each dataset sperately


- **main function**
> - DTi2Vec_main.py


---
## Installing:

To get the development environment runining, the code get 3 parameters from the user which are:
- **the dataset name** data:(nr, gpcr, ic, e, DrugBank)
- **the boosting classifier** classifier: AdaBoost(ab), XGBoost (xgbc)
- **the fusion function** func: (Concat, Hadmard, AVG, WL1, WL2)
- (the defual values are:  dataset:nr , classifier:ab, fusion function: )

- to run the code (to obtain best results for each dataset run the following:

```
python DTi2Vec_main.py --data nr --classifier ab --func WL1
```
```
python DTi2Vec_main.py --data gpcr --classifier xgbc --func Hadamard
```
```
python DTi2Vec_main.py --data ic --classifier xgbc --func Concat
```
```
python DTi2Vec_main.py --data e --classifier xgbc --func Concat
```
```
python DTi2Vec_main.py --data DrugBank --classifier xgbc --func Hadamard
```

------------------
### For citation:
---


