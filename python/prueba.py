## We set the chosen hyper-parameters
import stability as st
import pandas as pd
import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt

def getMutualInfos(data,labels):
    '''
    This function takes as input the data and labels and returns the mutual information of each feature 
    with the labels in a np.dnarray of length d
    
    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data
    
    OUPUT:
    - a 1-dimensional numpy.ndarray of length d (where d is the number of features) 
      with the mutual information of each feature with the label
    '''
    M,d=data.shape
    mutualInfos=np.zeros(d)
    # for each feature
    for f in range(d):
        # we calculate the mutual information of the feature with the labels
        mutualInfos[f]=metrics.mutual_info_score(data[:,f],labels)
    return mutualInfos


def getBootstrapSample(data,labels):
    '''
    This function takes as input the data and labels and returns 
    a bootstrap sample of the data, as well as its out-of-bag (OOB) data
    
    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data
    
    OUPUT:
    - a dictionnary where:
          - key 'bootData' gives a 2-dimensional numpy.ndarray which is a bootstrap sample of data
          - key 'bootLabels' is a 1-dimansional numpy.ndarray giving the label of each example in bootData
          - key 'OOBData' gives a 2-dimensional numpy.ndarray the OOB examples
          - key 'OOBLabels' is a 1-dimansional numpy.ndarray giving the label of each example in OOBData
    '''
    data = data.to_numpy()
    m,d=data.shape
    if m!= len(labels):
        raise ValueError('The data and labels should have a same number of rows.')
    ind=np.random.choice(range(m), size=m, replace=True)
    OOBind=np.setdiff1d(range(m),ind, assume_unique=True)
    bootData=data[ind,] #taking the ind row, all the columns
    bootLabels=labels[ind]
    OOBData=data[OOBind,]
    OOBLabels=labels[OOBind]
    return {'bootData':bootData,'bootLabels':bootLabels,'OOBData':OOBData,'OOBLabels':OOBLabels}
M=100
dataName='bbdd_hepatitis.csv'
## We load the desired dataset

res=pd.read_csv(dataName)
data=res.drop('label', axis = 1)
labels=res['label']
m,d=data.shape
kValues=range(1,d-1) ## an array with the number of features we want to select
labels=labels.values.reshape(m)
Z=np.zeros((len(kValues),M,d),dtype=np.int8) # this will store the M feature sets
for i in range(M):
    newData=getBootstrapSample(data,labels) ## we get bootstrap samples
    mutualInfos=getMutualInfos(newData['bootData'],newData['bootLabels']) ## we get the mutual informaion on the bootstrap samples
    ind=np.argsort(mutualInfos) #sorts the mutual informations in increasing order
    for j in range(len(kValues)):
        ## we retrieve the indices of the top-k mutual informations
        #print(list(range(d-1,d-1-kValues[j],-1)))
        topK=ind[range(d-1,d-1-kValues[j],-1)] ## the indices of the features with the kValues[k] highest MIs 
        Z[j,i,topK]=1 

# now we get the stability with confidence intervals   
    
stabilities=np.zeros(len(kValues))
stabErr=np.zeros(len(kValues))
for j in range(len(kValues)):
    res=st.confidenceIntervals(Z[j,],alpha=0.05)
    stabilities[j]=res['stability']
    stabErr[j]=stabilities[j]-res['lower']

plt.figure()
plt.errorbar(kValues,stabilities,yerr=stabErr,marker='o', linestyle='--')
plt.ylabel('stability')
plt.xlabel('Number of features selected k')
plt.title("Stability of the selection of the top-k feature against k on the "+dataName+" dataset")
plt.show()
