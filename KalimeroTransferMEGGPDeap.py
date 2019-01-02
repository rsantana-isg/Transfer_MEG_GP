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
  test_indices = range(0,ncases)

  
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

  classes_train = classes[0:ncases]
  classes_test = classes[0:ncases]

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





#####################################################################################################

def Pop_Effic_evalMEG_All(pop,psize):
 
 all_results = np.zeros((psize,16))
 # Evaluate the sum of correctly identified cases in the data set
 TrainV = [ [] for j in range(16) ]
  
 aux_data = loadmat('matlab_data/IndexMeanSlopeData.mat', squeeze_me=True) 
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
                                             verbose=1)      
    

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
    #np.random.seed(seed)
    random.seed(seed)



    nsel = int(number_features/2)
    bi_objective_functions = {0 : eval_NormalAcc_VarSim,    # Alg1
                              1 : eval_BiasAcc_VarSim,      # Alg2
                              2 : eval_NormalAcc_BiasAcc,   # Alg3 
                              3 : eval_Acc_LogisticRegression, # Alg4
                              8 : eval_BiasAcc_LogisticRegression,  # Alg5
                             }    
    
    #print(subject,target_subj,npop,ngen,type_class)
    All_ys = InitData()
    MEG_data_train,MEG_data_test,classes_train,classes_test = Read_InputData(subject,target_subj)
   
    toolbox = Init_GP_MOP(subject)

    for run in range(number_runs):
      pop, stats, hof = Apply_GP_MOP(toolbox,npop,ngen,run)
      for i in range(npop):
        print("VALS ",i, pop[i].fitness.values[0],pop[i].fitness.values[1])   
        print("PROG ",pop[i])
       


# EXAMPLE OF HOW TO CALL THE PROGRAM
#  ./KalimeroTransferMEGGPDeap.py 100 120 3 1 2 1000 100 1
