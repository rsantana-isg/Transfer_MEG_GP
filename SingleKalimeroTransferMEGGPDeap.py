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

def Read_InputData(subj):
  global SymMean 
  global SymSlope 
  global MatSymMean 
  global MatSymSlope 


  class_file = 'data/MEG_classes%d.csv' %(subj-1)  
  classes = np.loadtxt(class_file, delimiter=' ',unpack=True).astype(int) 
 
  ncases = classes.shape[0] 
  #training_indices = range(0,ncases,2)
  #test_indices = range(1,ncases,2)

  #classes_train = classes[0:ncases:2]
  #classes_test = classes[1:ncases:2]

  training_indices = range(0,ncases)
  test_indices = range(0,ncases)

  classes_train = classes[0:ncases]
  classes_test = classes[0:ncases]

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

  MEG_data_train = list(list(float(elem) for elem in row) for row in MEGReader[training_indices,:])  
  MEG_data_test = list(list(float(elem) for elem in row) for row in MEGReader[test_indices,:]) 
  
 
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
         meg_file = 'data/CMEGdataMean_%d.csv' %(j+1)
         auxMEGReader_mean = np.loadtxt(meg_file, delimiter=' ',usecols=meanSelVars[subject-1,:nsel]-1,unpack=True).astype(float) 
         meg_file = 'data/CMEGdataSlope_%d.csv' %(j+1)
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


####################################################################################################

#  Single-objective function
# It evaluates the accuracy of the genetic program as a classifier on the  source subject

def Single_eval_NormalAcc_VarSim(individual):
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
     
      
   
    return (result/ncases,)




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
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=2) # IT MIGHT BE A BUG WITH THIS
    #toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    toolbox.register("evaluate", Single_eval_NormalAcc_VarSim)
    #toolbox.register("evaluate",bi_objective_functions[type_function])

    toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selNSGA2)
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
    #np.random.seed(seed)
    random.seed(seed)



    nsel = int(number_features/2)
    
    All_ys = InitData()
    MEG_data_train,MEG_data_test,classes_train,classes_test = Read_InputData(subject)
   
    toolbox = Init_GP_MOP(subject)

    print("Subject:",subject," Population size: ",npop,"Number generations: ",ngen, "Number runs: ",number_runs)
    for run in range(number_runs):
      print("Run: ",run) 
      pop, stats, hof = Apply_GP_MOP(toolbox,npop,ngen,run)
      for i in range(npop):
        print("VALS ",i, 0.0, pop[i].fitness.values[0])   
        print("PROG ",pop[i])
       


# EXAMPLE OF HOW TO CALL THE PROGRAM
#  ./SingleKalimeroTransferMEGGPDeap.py 100 120 0 16 16 500 20
