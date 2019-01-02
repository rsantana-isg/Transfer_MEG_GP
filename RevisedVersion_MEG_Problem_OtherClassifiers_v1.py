#!/usr/bin/env python3

#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import random
import operator
import csv
import itertools
import numpy as np
#import pylab as pl
import tkinter
import matplotlib
#matplotlib.use('Qt4Agg') # interactive plotting backend
#print(matplotlib.rcParams['backend'])
import matplotlib.pylab as pl

import pylab
#import pygraphviz as pgv
from scipy.io import loadmat

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA     # sklearn.lda 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # sklearn.qda
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn import decomposition 
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier
from scipy.io import loadmat
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn import metrics



#from iw import ImportanceWeightedClassifier
from libtlda.iw import ImportanceWeightedClassifier




SymMean = []
SymSlope = []
MatSymMean = []
MatSymSlope = []


##########################################################################################################
# FUNCTIONS THAT READ THE DATA TO BE CLASSIFIED AND THE LABELS FOR ALL  SUBJECTS
##########################################################################################################



#####################################################################################################

# InitData reads a the file TrainingLabels.mat that contain the labels (0 or 1)
# for all the cases of each of the 16 subjects. This information is contained
# in the variable All_ys, where All_ys{i} is the vector containing the labels
# for the cases of subject i. E.g. All_ys{0}(1:10)' = [0,0,1,1,0,0,0,1,0,1], i.e.,
# the labels to the first 10 cases. Each subject has a different number of case.
def InitData():
   data = loadmat('matlab_data/TrainingLabels.mat', squeeze_me=True)
   All_ys = data['All_ys']
   return All_ys



#####################################################################################################

def Read_InputData(subj,tsubj):
  global SymMean 
  global SymSlope 
  global MatSymMean 
  global MatSymSlope 


  class_file = 'data/MEG_classes%d.csv' %(subj-1)  
  classes = np.loadtxt(class_file, delimiter=' ',unpack=True).astype(int)  
  ncases = classes.shape[0] 
  training_indices = range(0,ncases)
  classes_train = classes[0:ncases]

  class_file = 'data/MEG_classes%d.csv' %(tsubj-1)  
  classes = np.loadtxt(class_file, delimiter=' ',unpack=True).astype(int)  
  n_testcases = classes.shape[0] 
  test_indices = range(0,n_testcases)
  classes_test = classes[0:n_testcases]

  
  aux_data = loadmat('matlab_data/IndexMeanSlopeData.mat', squeeze_me=True) 
  meanSelVars  = aux_data['MeanVarsIndex']  
  slopeSelVars  = aux_data['SlopeVarsIndex']  

 
  #aux_data = loadmat('matlab_data/SymMeanSlope_gauss.mat', squeeze_me=True) 
  matlab_file = 'matlab_data/SymMeanSlope_rulif_%d.mat' %(subj)
  aux_data = loadmat(matlab_file, squeeze_me=True) 
  SymMean  = aux_data['SymMean']    
  SymSlope  = aux_data['SymSlope']  
  MatSymMean  = aux_data['MatSymMean']    
  MatSymSlope  = aux_data['MatSymSlope']  
  
 
 # print(MatSymMean[1].shape,nsel)
 # Reads the file with the features and the classes of the problem




  meg_file = 'data/CMEGdataMean_%d.csv' %(subj)
  auxMEGReader_mean = np.loadtxt(meg_file, delimiter=' ',usecols=meanSelVars[subj-1,:nsel]-1,unpack=True).astype(float) 
  meg_file = 'data/CMEGdataSlope_%d.csv' %(subj)
  auxMEGReader_slope = np.loadtxt(meg_file, delimiter=' ',usecols=slopeSelVars[subj-1,:nsel]-1,unpack=True).astype(float) 

  MEGReader = np.hstack((auxMEGReader_mean.transpose(),auxMEGReader_slope.transpose()))
  MEG_data_train = list(list(float(elem) for elem in row) for row in MEGReader[:,:])  
  

  meg_file = 'data/CMEGdataMean_%d.csv' %(tsubj)
  auxMEGReader_mean = np.loadtxt(meg_file, delimiter=' ',usecols=meanSelVars[subj-1,:nsel]-1,unpack=True).astype(float) 
  meg_file = 'data/CMEGdataSlope_%d.csv' %(tsubj)
  auxMEGReader_slope = np.loadtxt(meg_file, delimiter=' ',usecols=slopeSelVars[subj-1,:nsel]-1,unpack=True).astype(float) 
  
  MEGReader = np.hstack((auxMEGReader_mean.transpose(),auxMEGReader_slope.transpose()))
  MEG_data_test = list(list(float(elem) for elem in row) for row in MEGReader[:,:]) 
  
 
  return MEG_data_train,MEG_data_test,classes_train,classes_test




##########################################################################################################
# FUNCTIONS THAT IMPLEMENT TRADITIONAL CLASSIFIERS USING SCI-KIT LEARN
##########################################################################################################



def dist(x,y):   
    return np.sqrt(np.mean((x-y)**2))


def CrossValAnalysisProb(myclf,X,y):
    cv = StratifiedKFold(y, n_folds=5,shuffle=True,random_state=seed)  # For Python 
    res = y*-1.0
    for i, (train, test) in enumerate(cv):   
        #probas_ = myclf.fit(X[train], y[train],sample_weight=Limo_sample_weight[train]).predict(X[test])
        probas_ = myclf.fit(X[train], y[train]).predict_proba(X[test])
        res[test] = probas_[:,1]
    return res

def CrossValAnalysis(myclf,X,y):
    cv = StratifiedKFold(y, n_folds=5,shuffle=True,random_state=seed)  # For Python 
    res = y*-1.0
    for i, (train, test) in enumerate(cv):       
        #probas_ = myclf.fit(X[train], y[train],sample_weight=Limo_sample_weight[train]).predict(X[test])
        probas_ = myclf.fit(X[train], y[train]).predict(X[test])
        res[test] = probas_
    return res

def ErrorAnalysis(myclf,X,y):
# Run myclf with crossvalidation 
    cv = KFold(y.shape[0], n_folds=10)
    res =  np.zeros([X.shape[0]])
    for i, (train, test) in enumerate(cv):       
        probas_ = myclf.fit(X[train], y[train]).predict(X[test])
        #print probas_.shape, res.shape, train.shape, test.shape
        res[test] = probas_           
    
    return res


def Weighted_InitClassifier(index):
   C = 1.0
   importance = range(1,10)
   parameters1 = {'kernel': ['linear'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],'C': [1, 10, 100, 1000]}
   parameters2 = {'kernel': ['poly'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
   parameters3 = {'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
 
   if index==1:      
     clf=LogisticRegression(C=C, penalty='l1')
   if index==2:  
     clf=LogisticRegression(C=C, penalty='l2')
   if index==3:  
     clf=QDA()
   if index==4:
     clf=LDA()             
   if index==5:  
     clf=KNeighborsClassifier()
   if index==6:  
     clf=SVC(kernel='linear', C=C, probability=True, tol=1e-3, verbose=False)        
   if index==7:
     clf = svm.SVC(kernel='poly', degree=3, C=C, probability=True, tol=1e-4, verbose=False)
   if index==8:
     clf = svm.SVC(kernel='rbf', C=C, probability=True, tol=1e-4, verbose=False)
   if index==9:
     clf = GaussianNB() 
   if index==10:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=11, subsample=1.0)    
   if index==11:
     clf = RandomForestClassifier(max_depth=11, n_estimators=100)    
   if index==12:
     clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1.0,random_state=0)
   if index==13:
     clf = ExtraTreesClassifier(n_estimators=100,random_state=0)     

  

   #if index==55:
   #  clf = RandomizedLogisticRegression() # This
   return clf



 
def GridSearch_Weighted_InitClassifier(index,sw):
   C = 1.0
   importance = range(1,10)
   parameters1 = {'kernel': ['linear'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],'C': [1, 10, 100, 1000]}
   parameters2 = {'kernel': ['poly'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
   parameters3 = {'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}

   grid_logistic = {"C":np.logspace(-3,3,16)}
   Cs = [0.001, 0.01, 0.1, 1, 10]
   gammas = [0.001, 0.01, 0.1, 1]
   grid_SVMs = {'C': Cs, 'gamma' : gammas}
   grid_GB = {'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

   grid_gradient_boosting = {'learning_rate': [0.1, 0.05, 0.02],
              'max_depth': [6, 8, 11],
              'min_samples_leaf': [20, 50,100]
              }

   grid_RF = {
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200]
    }

   grid_DT = {
    'criterion': ['gini','entropy'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    }
 
   
   fit_params = {'sample_weight': sw}

   
   if index==1:      
     logreg = LogisticRegression(penalty='l1')
     clf = GridSearchCV(logreg,grid_logistic,cv=5, fit_params=fit_params)
   if index==2:  
     logreg = LogisticRegression(penalty='l2')
     clf = GridSearchCV(logreg,grid_logistic,cv=5, fit_params=fit_params)
   if index==3:  
     clf=QDA()
   if index==4:
     clf=LDA()             
   if index==5:  
     clf=KNeighborsClassifier()
   if index==6:  
     aux_clf = SVC(kernel='linear', probability=True, tol=1e-3, verbose=False)
     clf  = GridSearchCV(aux_clf,grid_SVMs,cv=5, fit_params=fit_params)     
   if index==7:
     aux_clf = SVC(kernel='poly', degree=3, probability=True, tol=1e-4, verbose=False)
     clf  = GridSearchCV(aux_clf,grid_SVMs,cv=5, fit_params=fit_params)   
   if index==8:
     aux_clf = SVC(kernel='rbf', probability=True, tol=1e-4, verbose=False)
     clf  = GridSearchCV(aux_clf,grid_SVMs,cv=5, fit_params=fit_params)  
   if index==9:
     clf = GaussianNB()        
   if index==10:
     aux_clf = GradientBoostingClassifier(n_estimators=100, subsample=1.0)       
     clf  = GridSearchCV(aux_clf,grid_gradient_boosting,cv=5, fit_params=fit_params)        
   if index==11:
     aux_clf = RandomForestClassifier()    
     clf  = GridSearchCV(aux_clf,grid_RF,cv=5, fit_params=fit_params)  
   if index==12:
     aux_clf = DecisionTreeClassifier(max_depth=None,random_state=0)
     clf  = GridSearchCV(aux_clf,grid_DT,cv=5, fit_params=fit_params)  
   if index==13:
     aux_clf = ExtraTreesClassifier(random_state=0)
     clf  = GridSearchCV(aux_clf,grid_DT,cv=5, fit_params=fit_params)  

  

   #if index==55:
   #  clf = RandomizedLogisticRegression() # This
   return clf

def InitClassifier(index):
   C = 1.0
   importance = range(1,10)
   parameters1 = {'kernel': ['linear'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],'C': [1, 10, 100, 1000]}
   parameters2 = {'kernel': ['poly'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
   parameters3 = {'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}

   if index==1:      
     clf=LogisticRegression(C=C, penalty='l1')
   if index==2:  
     clf=LogisticRegression(C=C, penalty='l2')
   if index==3:  
     clf=QDA()
   if index==4:
     clf=LDA()             
   if index==5:  
     clf=KNeighborsClassifier()
   if index==6:  
     clf=SVC(kernel='linear', C=C, probability=True, tol=1e-3, verbose=False)        
   if index==7:
     clf = svm.SVC(kernel='poly', degree=3, C=C, probability=True, tol=1e-4, verbose=False)
   if index==8:
     clf = svm.SVC(kernel='rbf', C=C, probability=True, tol=1e-4, verbose=False)
   if index==9:
     clf = GaussianNB() 
   if index==10:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=11, subsample=1.0)    
   if index==11:
     clf = RandomForestClassifier(max_depth=11, n_estimators=100)    
   if index==12:
     clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1.0,random_state=0)     
   if index==13:
     clf = ExtraTreesClassifier(n_estimators=100,random_state=0)     
   if index==14:
     clf = DecisionTreeClassifier(max_depth=5)   
   if index==15:
     clf = RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1)     
   if index==16:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=11)     
   if index==17:
     clf = linear_model.SGDClassifier(loss='log')
   if index==18:  
     clf=KNeighborsClassifier(n_neighbors=5)
   if index==19:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=3)   
   if index==20:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=5)   
   if index==21:
     clf = RandomForestClassifier(max_depth=11, n_estimators=500)  
   if index==22:
     clf = RandomForestClassifier(max_depth=13, n_estimators=700)   
   if index==23:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=11)   # 1000 for 23,24,25,26
   if index==24:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=12)   
   if index==25:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=13)   
   if index==26:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=14)   
   if index==27:
     clf = linear_model.SGDClassifier(loss='log', penalty='l1') 
   if index==28:
     clf = linear_model.SGDClassifier(loss='log', penalty='l2' )     
   if index==29:
     clf = linear_model.SGDClassifier(loss='modified_huber',penalty='l1') 
   if index==30:
     clf = linear_model.SGDClassifier(loss='modified_huber',penalty='l2') 
   if index==31:
     clf = Perceptron() 
   if index==32:
     clf = GridSearchCV(SVC(kernel='linear',verbose=False),parameters1,verbose=False)
   if index==33:
     clf = linear_model.SGDClassifier(loss='hinge') 
   if index==34:
     clf = NearestCentroid()
   if index==35:
     clf = RandomForestClassifier(max_depth=5, n_estimators=50)    
   if index==36:
     clf = GridSearchCV(SVC(kernel='poly',verbose=False),parameters2,verbose=False)
   if index==37:
     clf = GridSearchCV(SVC(kernel='rbf',verbose=False),parameters3,verbose=False)
   if index==38:
     clf = NearestCentroid(metric='l1')
   if index==39:
     clf = NearestCentroid(metric='l2')
   if index==40:
     clf = NearestCentroid(metric='correlation')
   if index==41:
     clf = NearestCentroid(metric='cosine')
   if index==42:
     clf = NearestCentroid(metric='seuclidean')
   if index==43:
     clf = GradientBoostingClassifier(n_estimators=25, max_depth=5, subsample=1.0)  # This
   if index==44:
     clf = GradientBoostingClassifier(n_estimators=30, max_depth=5, subsample=1.0) 
   if index==45:
     clf = GradientBoostingClassifier(n_estimators=40, max_depth=5, subsample=1.0) 
   if index==46:
     clf = RandomForestClassifier(max_depth=7, n_estimators=80)
   if index==47:
     clf = RandomForestClassifier(max_depth=7, n_estimators=100)
   if index==48:
     clf = RandomForestClassifier(max_depth=7, n_estimators=120)
   if index==49:
     clf = RandomForestClassifier(max_depth=9, n_estimators=150) # This
   if index==50:
     clf = RandomForestClassifier(max_depth=5, n_estimators=10)
   if index==51:
     clf = RandomForestClassifier(max_depth=5, n_estimators=15)
   if index==52:
     clf = RandomForestClassifier(max_depth=9, n_estimators=10)
   if index==53:
     clf = RandomForestClassifier(max_depth=11, n_estimators=10) # This
   if index==54:
     clf = RandomForestClassifier(max_depth=11, n_estimators=15) # This
   if index==55:
     clf = DecisionTreeClassifier(max_depth=7)   
   if index==56:
     clf = DecisionTreeClassifier(max_depth=10)   
   if index==57:
     clf = DecisionTreeClassifier(max_depth=15)   
   if index==58:
     clf = DecisionTreeClassifier(max_depth=20)   

   #if index==55:
   #  clf = RandomizedLogisticRegression() # This
   return clf




#####################################################################################################



def InitTransferClassifier(index,l,iw):
   if index==1:      
     clf = ImportanceWeightedClassifier(loss=l,iwe=iw)
   if index==2:  
     clf = TransferComponentClassifier()
   if index==3:  
     clf=SubspaceAlignedClassifier()
   if index==4:
     clf=StructuralCorrespondenceClassifier()             
   if index==5:  
     clf=RobustBiasAwareClassifier()
   if index==6:  
     clf=FeatureLevelDomainAdaptiveClassifier()
   if index==7:  
     clf=TargetContrastivePessimisticClassifier()        
  
   return clf

#####################################################################################################

def Pop_Effic_evalMEG_All(pop,psize):
 
 all_results = np.zeros((psize,16))
 # Evaluate the sum of correctly identified cases in the data set
 TrainV = [ [] for j in range(16) ]
  
 aux_data = loadmat('../matlab_data/IndexMeanSlopeData.mat', squeeze_me=True) 
 meanSelVars  = aux_data['MeanVarsIndex']  
 slopeSelVars  = aux_data['SlopeVarsIndex']  
 for j in range(16):   
      if (j+1)!=subject:
         meg_file = 'CMEGdataMean_%d.csv' %(j+1)
         auxMEGReader_mean = np.loadtxt(meg_file, delimiter=' ',usecols=meanSelVars[subject-1,:nsel]-1,unpack=True).astype(float) 
         meg_file = 'CMEGdataSlope_%d.csv' %(j+1)
         auxMEGReader_slope = np.loadtxt(meg_file, delimiter=' ',usecols=slopeSelVars[subject-1,:nsel]-1,unpack=True).astype(float) 
         X = np.hstack((auxMEGReader_mean.transpose(),auxMEGReader_slope.transpose()))    
         y = All_ys[j]
      else:
         X =  MEG_data_test
         y = classes_test
      
      ncases = len(y)      
      for l in range(psize):    
         individual = pop[l]
         # Transform the tree expression in a callable function
         func = toolbox.compile(expr=individual)  
         result = 0       
         for i,cases in enumerate(X):
           aux = func(*cases)
           tot = (aux==bool(y[i]))
           result = result + tot             
         all_results[l,j] = result/ncases
 #     print(l,j, result/ncases)
 return all_results






#####################################################################################################

# Compute_Frequencies computes the frequencies of each non-terminal in
# the tree program

def Compute_Frequencies(individual):
    # Transform the tree expression in a callable function
    auxp = str(individual)
    prog_list = auxp.split("IN")    
    Frequencies = np.zeros((2*nsel))
    #print(prog_list)
    for k in range(0,len(prog_list)):
           try:
             x = int(prog_list[k][:3])             
           except ValueError:
             try:
                x = int(prog_list[k][:2])                
             except ValueError:
                try:
                   x = int(prog_list[k][:1])
                except ValueError:
                   x = -1
           #print(k,x)
           if x>-1:
               Frequencies[x] = Frequencies[x] + 1
    return Frequencies



#####################################################################################################

#  is the bi-objective function used by the transfer GP algorithm
# It evaluates the accuracy of the genetic program that serves as a classifier and similarity
# between the terminals used by source and target

def eval_NormalAcc_VarSim(individual):
    # Transform the tree expression in a callable function
    
    MatMean = np.asarray(MatSymMean[target_subj-1])
    MatSlope = np.asarray(MatSymSlope[target_subj-1])   

    Frequencies =  Compute_Frequencies(individual)             
    func = toolbox.compile(expr=individual)

    # Evaluate the sum of correctly identified cases in the data set
    result = 0
    ncases = len(classes_train)
    
    # If there is not at least one not-terminal the tree is penalized
    posfreq_mean  = np.asarray(np.where(Frequencies[:nsel]>0)[0]);
    posfreq_slope  = np.asarray(np.where(Frequencies[nsel:]>0)[0]);    
    
    #print(Frequencies,posfreq_mean.shape[0],nsel)
    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:  
     for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))*1.0                   
       result = result + tot
       #print(i,aux, classes_train[i],tot,result) 
    else:
       result = 0
     
    
    # We measure the si  
    t_tot = 0;   
    for i in range(nsel):
     if Frequencies[i]>0:
      t_tot = t_tot+SymMean[target_subj-1,i]     
     if Frequencies[i+nsel]>0:
      t_tot = t_tot+SymSlope[target_subj-1,i]      
   
   
    return (-1*t_tot,result/ncases)
   


#####################################################################################################

def eval_NormalAcc_BiasAcc(individual):
    # Transform the tree expression in a callable function
    
    MatMean = np.asarray(MatSymMean[target_subj-1])
    MatSlope = np.asarray(MatSymSlope[target_subj-1])   
  
    Frequencies =  Compute_Frequencies(individual)             
    func = toolbox.compile(expr=individual)

    # Evaluate the sum of correctly identified cases in the data set
    result = 0
    bias_result = 0
    ncases = len(classes_train)
    posfreq_mean  = np.asarray(np.where(Frequencies[:nsel]>0)[0]);
    posfreq_slope  = np.asarray(np.where(Frequencies[nsel:]>0)[0]);    

    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:  
     for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))             
       importance_mean = np.mean(MatSymMean[target_subj-1][posfreq_mean,i]) 
       importance_slope = np.mean(MatSymSlope[target_subj-1][posfreq_slope,i])   
       importance = np.mean([importance_mean,importance_slope])      
       result = result + tot
       bias_result = bias_result + tot * importance
    else:
       result = 0
       bias_result = 0
         
    return (bias_result/ncases,result/ncases)
    



#####################################################################################################
def eval_BiasAcc_VarSim(individual):
    # Transform the tree expression in a callable function
    
    MatMean = np.asarray(MatSymMean[target_subj-1])
    MatSlope = np.asarray(MatSymSlope[target_subj-1])   
  
    Frequencies =  Compute_Frequencies(individual)             
    func = toolbox.compile(expr=individual)

    # Evaluate the sum of correctly identified cases in the data set
    bias_result = 0
    ncases = len(classes_train)
    posfreq_mean  = np.asarray(np.where(Frequencies[:nsel]>0)[0]);
    posfreq_slope  = np.asarray(np.where(Frequencies[nsel:]>0)[0]);    

    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:  
     for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))      
       
       importance_mean = np.mean(MatSymMean[target_subj-1][posfreq_mean,i]) 
       importance_slope = np.mean(MatSymSlope[target_subj-1][posfreq_slope,i])   
       importance = np.mean([importance_mean,importance_slope])
       #print(i,posfreq_mean,posfreq_slope,importance)      
      
       bias_result = bias_result + tot * importance
    else:
       bias_result = 0
    t_tot = 0;
    n_cas = 1

    for i in range(nsel):
     if Frequencies[i]>0:
      t_tot = t_tot+SymMean[target_subj-1,i]     
     if Frequencies[i+nsel]>0:
      t_tot = t_tot+SymSlope[target_subj-1,i]           
    
    return (-1*t_tot,bias_result/ncases)
  



#####################################################################################################

# eval_Acc_Logistics is a  bi-objective function used by the transfer GP algorithm
# One objectives  evaluates the accuracy of the genetic program that serves as a classifier
# The second objectives evaluates how good is the set of terminals included in the genetic program
# to discriminate between samples in the train set (source) and the test set (target). 
# The discrimination capacity of the features is evaluated using a logistic regression classifier
# to distinguish between source and target

def eval_Acc_LogisticRegression(individual):
    # Transform the tree expression in a callable function
    
    MatMean = np.asarray(MatSymMean[target_subj-1])
    MatSlope = np.asarray(MatSymSlope[target_subj-1])   

    Frequencies =  Compute_Frequencies(individual)             
   

    # Evaluate the sum of correctly identified cases in the data set
    result = 0
    ncases = len(classes_train)
    
    # If there is not at least one not-terminal the tree is penalized
    posfreq_mean  = np.asarray(np.where(Frequencies[:nsel]>0)[0]);
    posfreq_slope  = np.asarray(np.where(Frequencies[nsel:]>0)[0]);    
    
    #print(posfreq_mean.shape[0],posfreq_slope.shape[0])


    
    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:  
     func = toolbox.compile(expr=individual) 
     for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))*1.0                   
       result = result + tot
       #print(i,aux, classes_train[i],tot,result) 
    else:
       result = 0
       LR_accuracy = 150
       return (-LR_accuracy,result/ncases)

      
    # Here the logistic classifier is used to measure the similarity
    
    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:   
       joint_features = np.hstack((posfreq_mean,nsel+posfreq_slope))
       #print(joint_features)
       AuxTrain = np.asarray(MEG_data_train)
       AuxTest = np.asarray(MEG_data_test)
       LR_training_data = np.vstack((AuxTrain[:,joint_features],AuxTest[:,joint_features[:]])) 
       LR_data_labels = np.vstack((np.ones((AuxTrain.shape[0],1)),np.zeros((AuxTest.shape[0],1)) ))    
     
      
       if type_class==0:
          clf = GaussianNB()
       elif type_class==1:
          clf = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced')     
       elif type_class==2:
          clf = RandomForestClassifier(max_depth=5, n_estimators=20)  

       clf.fit(LR_training_data,np.ravel(LR_data_labels)) # LR Learning
  
       aux_ys = clf.predict(LR_training_data)   # LR Prediction   
  
       #print(aux_ys.shape,LR_data_labels.shape)
       #acc = np.random.randint(100)
       acc = 0.0 
       for j in range(LR_data_labels.shape[0]):
           #print(j,LR_data_labels.shape[0],aux_ys[j],LR_data_labels[j],np.abs(aux_ys[j]-LR_data_labels[j]),acc)
           acc = acc + int(1.0-np.abs(aux_ys[j]-LR_data_labels[j]))              
       #acc = int(acc)
       LR_accuracy =  (100.0*acc)/LR_data_labels.shape[0]       
       #print(acc,LR_data_labels.shape[0],LR_accuracy,result/ncases)
       #print(acc,LR_accuracy)
   
    return (-LR_accuracy,result/ncases)



#####################################################################################################

# eval_BiasAcc_Logistics is a  bi-objective function used by the transfer GP algorithm
# One objectives  evaluates the accuracy of the genetic program that serves as a classifier
# the classifier is biased using information computed a priori
# The second objectives evaluates how good is the set of terminals included in the genetic program
# to discriminate between samples in the train set (source) and the test set (target). 
# The discrimination capacity of the features is evaluated using a classifier
# to distinguish between source and target

def eval_BiasAcc_LogisticRegression(individual):
   
    # Transform the tree expression in a callable function
    
    MatMean = np.asarray(MatSymMean[target_subj-1])
    MatSlope = np.asarray(MatSymSlope[target_subj-1])   

    Frequencies =  Compute_Frequencies(individual)             
   
    # Evaluate the sum of correctly identified cases in the data set
    bias_result = 0
    ncases = len(classes_train)
   
    posfreq_mean  = np.asarray(np.where(Frequencies[:nsel]>0)[0]);
    posfreq_slope  = np.asarray(np.where(Frequencies[nsel:]>0)[0]);        
    
    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:  
     func = toolbox.compile(expr=individual) 
     for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))             
       importance_mean = np.mean(MatSymMean[target_subj-1][posfreq_mean,i]) 
       importance_slope = np.mean(MatSymSlope[target_subj-1][posfreq_slope,i])   
       importance = np.mean([importance_mean,importance_slope])   
       bias_result = bias_result + tot * importance
    else:
       bias_result = 0
       LR_accuracy = 150
       return (-LR_accuracy,bias_result/ncases)
      
    # Here the logistic classifier is used to measure the similarity
    
    if posfreq_mean.shape[0]> 0 and posfreq_slope.shape[0]>0:   
       joint_features = np.hstack((posfreq_mean,nsel+posfreq_slope))
       #print(joint_features)
       AuxTrain = np.asarray(MEG_data_train)
       AuxTest = np.asarray(MEG_data_test)
       LR_training_data = np.vstack((AuxTrain[:,joint_features],AuxTest[:,joint_features[:]])) 
       LR_data_labels = np.vstack((np.ones((AuxTrain.shape[0],1)),np.zeros((AuxTest.shape[0],1)) ))         
      
       if type_class==0:
          clf = GaussianNB()
       elif type_class==1:
          clf = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced')     
       elif type_class==2:
          clf = RandomForestClassifier(max_depth=5, n_estimators=20)  

       clf.fit(LR_training_data,np.ravel(LR_data_labels)) # LR Learning  
       aux_ys = clf.predict(LR_training_data)   # LR Prediction     
       acc = 0.0 
       for j in range(LR_data_labels.shape[0]):  
           acc = acc + int(1.0-np.abs(aux_ys[j]-LR_data_labels[j]))              

       LR_accuracy =  (100.0*acc)/LR_data_labels.shape[0]       
   
    return (-LR_accuracy,bias_result/ncases)




#####################################################################################################

# eval_Biv_Transfer_Freq is the bi-objective function used by the transfer GP algorithm
# It evaluates the accuracy of the genetic program that serves as a classifier and similarity
# between the terminals used by source and target including the frequency of these terminals
# as a weight. Those terminals that are more frequent gets a higher weight in the function


def eval_Biv_Transfer_Freq(individual):
    # Transform the tree expression in a callable function
    auxp = str(individual)
    prog_list = auxp.split("IN")
    Frequencies = np.zeros((120))
    for k in range(0,len(prog_list)):
           try:
             x = int(prog_list[k][:3])             
           except ValueError:
             try:
                x = int(prog_list[k][:2])                
             except ValueError:
                try:
                   x = int(prog_list[k][:1])
                except ValueError:
                   x = -1
           #print(k,x)
           if x>-1:
               Frequencies[x] = Frequencies[x] + 1
    func = toolbox.compile(expr=individual)
    # Evaluate the sum of correctly identified cases in the data set
    result = 0
    ncases = len(classes_train)
    for i,cases in enumerate(MEG_data_train):
       aux = func(*cases)
       tot = (aux==bool(classes_train[i]))
       result = result + tot
    t_tot = 0; 
    n_cas = 1

    for i in range(nsel):
     
     if Frequencies[i]>0:
      t_tot = t_tot+SymMean[subject-1][target_subj-1,i]*Frequencies[i]
      #n_cas = n_cas + Frequencies[i]
     if Frequencies[i+nsel]>0:
      t_tot = t_tot+SymSlope[subject-1][target_subj-1,i]*Frequencies[i+nsel]  
      #n_cas = n_cas + Frequencies[i+nsel]
 
      #print (i,sum(Frequencies),t_tot,t_tot/(sum(Frequencies)+1))
   
    return (t_tot,result/ncases)
    #return (t_tot/n_cas,result/ncases)

  


##################################################################################################
# DEFINITION AND IMPLEMENTATION OF THE GP PROGRAMS
##################################################################################################


def GP_Definitions(nfeatures):
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, nfeatures), bool, "IN")

    # boolean operators
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    # floating point operators
    # Define a safe division function
    def safeDiv(left, right):
        try: return left / right
        except ZeroDivisionError: return 0
    pset.addPrimitive(operator.add, [float,float], float)
    pset.addPrimitive(operator.sub, [float,float], float)
    pset.addPrimitive(operator.mul, [float,float], float)
    pset.addPrimitive(safeDiv, [float,float], float)

    # logic operators
    # Define a new if-then-else function
    def if_then_else(input, output1, output2):
        if input: return output1
        else: return output2

    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # terminals
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset


#####################################################################################################

# Initialization of the Multi-Objective GP

def Init_GP_MOP(subj):
    nfeat = number_features
    pset = GP_Definitions(nfeat)
    maxDepthLimit = 10
    creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=2) # IT MIGHT BE A BUG WITH THIS
    #toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    #toolbox.register("evaluate", eval_Biv_Transfer_Freq)
    toolbox.register("evaluate",bi_objective_functions[type_function])

    #toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate('mutate',gp.staticLimit(key=operator.attrgetter('height'),max_value=maxDepthLimit))
    toolbox.decorate('mate',gp.staticLimit(key=operator.attrgetter('height'),max_value=maxDepthLimit))

    return toolbox


#####################################################################################################

# Application of the multi-objective GP

def Apply_GP_MOP(toolbox,pop_size,gen_number,therun):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats = tools.Statistics()
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    #res, logbook = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, gen_number, stats, halloffame=hof,verbose=1)

    res, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, 
                                     lambda_=pop_size, 
                                     cxpb=1-0.1,
                                     mutpb=0.1, 
                                     stats=stats, 
                                     halloffame=hof,
                                     ngen=gen_number, 
                                             verbose=0)      
    

    return res, logbook, hof




##################################################################################################
#  MAIN PROGRAM
##################################################################################################

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(2000),
         nargs='+', help='an integer in the range 0..2000')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum,
        default=max, help='sum the integers (default: find the max)')


    # The parameters of the program are set or read from command line

    global Gen                  # Current generation
    

    number_runs = 1             # Number of times that the GP program is executed

    args = parser.parse_args()
    seed = args.integers[0]               # Seed: Used to set different outcomes of the stochastic program
    number_features =  args.integers[1]   # Number of features. It coincides with the number of terminals
                                          # of the program. The first nsel features are of one type (mean)
                                          # and the last nsel features are of another type (slope) but this
                                          # difference should be transparent to the program 

    type_function = args.integers[2]      # Type of learning.  It determines the strategy used for classification
                                          # and influences the method used to read the data and the algorithm
                                          # used to evaluate the GP programs.

    subject = args.integers[3]         # Source subject whose data is used to learn the GP classifier
    target_subj = args.integers[4]     # Target subject whose data is classified 
    npop = args.integers[5]            # Population size of the GP programs
    ngen = args.integers[6]            # Number of generations
    type_class = args.integers[7]      # Type of classifier for transfer: 0:GaussianNB, 1:Logistic_l1, 2:RF
    mode =  args.integers[8]           # Evaluation mode. Relevant for the Article is GridSearch (mode==3)
    #np.random.seed(seed)
    random.seed(seed)


    loss_functions = ['logistic','quadratic', 'hinge']
    weighting_functions = ['lr', 'nn', 'rg','kde']
    nsel = int(number_features/2)
    bi_objective_functions = {0 : eval_NormalAcc_VarSim,
                              1 : eval_BiasAcc_VarSim,
                              2 : eval_NormalAcc_BiasAcc,    
                              3 : eval_Acc_LogisticRegression,
                              8 : eval_BiasAcc_LogisticRegression,
                             }    
    
    #print(subject,target_subj,npop,ngen,type_class)
    All_ys = InitData()
    MEG_data_train,MEG_data_test,classes_train,classes_test = Read_InputData(subject,target_subj)
   

    if mode==0:            # Evaluation of classifiers using 5-fold cross-validation
       y = classes_train       
       nsamples = y.shape[0]
       for index in  [1,2,3,4,5,6,7,8,9,10,11,12,13]:  #range(1,n_classifiers):
            clf = InitClassifier(index)       
            X = np.asarray(MEG_data_train)
         
            #train_probas = CrossValAnalysisProb(clf,X,y)   # Analysis taking into account probabilities (only for classifiers>5)
            train_probas = CrossValAnalysis(clf,X,y)   
            accuracy_org = sum((train_probas>0.5)==y)/(1.0*nsamples)           
            print (subject,index,accuracy_org)
    elif mode==1:            # Evaluation of classifiers using the full training set as test set
       y = classes_train       
       nsamples = y.shape[0]
       n_test_samples = classes_test.shape[0]
       for index in  [1,2,3,4,5,6,7,8,9,10,11,12,13]:  #range(1,n_classifiers):
             clf = InitClassifier(index)  
             X = np.asarray(MEG_data_train)     
             #probas_ = clf.fit(X, y).predict_proba(X)[:,1]    # Analysis taking into account probabilities (only for classifiers>5)
             probas_ = clf.fit(X, y).predict(X)
             accuracy_org = sum((probas_>0.5)==y)/(1.0*nsamples)           
             Y = np.asarray(MEG_data_test)                
             target_probas_ = clf.predict(Y)
             #print(classes_test.shape,Y.shape,n_test_samples)
             accuracy_targ = sum((target_probas_>0.5)==classes_test)/(1.0*n_test_samples)           
             print (subject,target_subj,index,accuracy_org,accuracy_targ)
    elif mode==2:            # Evaluation of classifiers using the full training set as test set
       y = classes_train       
       nsamples = y.shape[0]
       n_test_samples = classes_test.shape[0]
       for index in [1]:
         for l in np.arange(3):        
            for iw  in np.arange(4):
               clf = InitTransferClassifier(index,loss_functions[l],weighting_functions[iw])
               X = np.asarray(MEG_data_train)     
               Y = np.asarray(MEG_data_test)             
               clf.fit(X,y,Y)
               probas_ = clf.predict(X)
               accuracy_org = sum((probas_>0.5)==y)/(1.0*nsamples)
               target_probas_ = clf.predict(Y)
               #print(classes_test.shape,Y.shape,n_test_samples)
               accuracy_targ = sum((target_probas_>0.5)==classes_test)/(1.0*n_test_samples)           
               print (subject,target_subj,index,l,iw,accuracy_org,accuracy_targ)

    elif mode==3:            # Evaluation of classifiers using the full training set as test set
       y = classes_train
       nsamples = y.shape[0]
       n_test_samples = classes_test.shape[0]
       for index in  [1,2,6,7,8,9,10,11,12,13]:  #range(1,n_classifiers):          
          for iw  in [0,1,2,3]:
             iwe = weighting_functions[iw] 
             w_clf = ImportanceWeightedClassifier(iwe=iwe)
             X = np.asarray(MEG_data_train)
             Y = np.asarray(MEG_data_test)
             w_clf.fit(X,y,Y)
             if iwe == 'lr':
                w = w_clf.iwe_logistic_discrimination(X, Y)
             elif iwe == 'rg':
                w = w_clf.iwe_ratio_gaussians(X, Y)
             elif iwe == 'nn':
                w = w_clf.iwe_nearest_neighbours(X, Y)
             elif iwe == 'kde':
                w = w_clf.iwe_kernel_densities(X, Y)
             elif iwe == 'kmm':
                w = w_clf.iwe_kernel_mean_matching(X, Y)
             else:
                raise NotImplementedError('Estimator not implemented.')

             clf = Weighted_InitClassifier(index)             
              # Find importance-weights            

             probas_ = clf.fit(X, y, sample_weight=w).predict(X)
             accuracy_org = sum((probas_>0.5)==y)/(1.0*nsamples)           
                         
             target_probas_ = clf.predict(Y)
             #print(classes_test.shape,Y.shape,n_test_samples)
             accuracy_targ = sum((target_probas_>0.5)==classes_test)/(1.0*n_test_samples)           
             print (subject,target_subj,index,iw,accuracy_org,accuracy_targ)
            
    elif mode==4:            # Evaluation of classifiers using the full training set as test set
        
       y = classes_train
       nsamples = y.shape[0]
       n_test_samples = classes_test.shape[0]
       for index in  [1,2,6,7,8,9,10,11,12,13]:  #range(1,n_classifiers):
       #for index in  [12,13]:  #range(1,n_classifiers):      
          for iw  in [0,1,2,3]:
             iwe = weighting_functions[iw] 
             w_clf = ImportanceWeightedClassifier(iwe=iwe)
             X = np.asarray(MEG_data_train)
             Y = np.asarray(MEG_data_test)
             w_clf.fit(X,y,Y)
             if iwe == 'lr':
                w = w_clf.iwe_logistic_discrimination(X, Y)
             elif iwe == 'rg':
                w = w_clf.iwe_ratio_gaussians(X, Y)
             elif iwe == 'nn':
                w = w_clf.iwe_nearest_neighbours(X, Y)
             elif iwe == 'kde':
                w = w_clf.iwe_kernel_densities(X, Y)
             elif iwe == 'kmm':
                w = w_clf.iwe_kernel_mean_matching(X, Y)
             else:
                raise NotImplementedError('Estimator not implemented.')             
             
             clf = GridSearch_Weighted_InitClassifier(index,w)             
              # Find importance-weights            

             #probas_ = clf.fit(X, y, sample_weight=w).predict(X)
             probas_ = clf.fit(X, y).predict(X)             
             accuracy_org = sum((probas_>0.5)==y)/(1.0*nsamples)           
                         
             target_probas_ = clf.predict(Y)
             #print(target_probas_[:15])
             #print(classes_test.shape,Y.shape,n_test_samples)
             accuracy_targ = sum((target_probas_>0.5)==classes_test)/(1.0*n_test_samples)           
             print (subject,target_subj,index,iw,accuracy_org,accuracy_targ)
         

      #outputfilename = 'Classifiers_Evaluation.csv'  
      #np.savetxt(outputfilename,new_pred, fmt='%4.4e',)


   
    #toolbox = Init_GP_MOP(subject)

    #for run in range(number_runs):
    #  pop, stats, hof = Apply_GP_MOP(toolbox,npop,ngen,run)
    #  for i in range(npop):
    #    print("VALS ",i, pop[i].fitness.values[0],pop[i].fitness.values[1])   
    #    print("PROG ",pop[i])
       


# EXAMPLE OF HOW TO CALL THE PROGRAM
#  ./RevisedVersion_MEG_Problem_OtherClassifiers_v1.py 100 120 3 1 2 1000 100 2 2
