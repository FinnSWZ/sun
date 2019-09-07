import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import initializers
import tensorflow as tf
from keras.models import load_model
import random
from DNN_model import DBN_model,make_kid
from read_wilt_data import read_wilt_data, wilt_fitness
from read_Balance_data import read_balance_data, balance_fitness
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

min_lr = 0.001
max_lr = 0.8
min_ep = 32
max_ep = 1000
min_bs = 16
max_bs = 256
min_node = 1
max_node = 128
N_generation = 50
mutation_strength = [0.3,200,50,30,30]

# limit parameter range
def limit(parameter,minimum,maximum):
    parameter = max(parameter,minimum)
    parameter = min(parameter,maximum)
    
    return parameter 
# initialize data
def initialization(parameter,max_bound, p_type=0):
    parameter = np.random.rand() * max_bound
    if p_type == 0:
        parameter =int(parameter)
    return parameter

def offspring(parent):
    for i in range(len(kid)):
        kid[i] = parent[i] + mutation_strength[i] * np.random.randn()
        if i == 0:
            kid[i] = limit(kid[i],min_lr,max_lr)
        elif i == 1:
            kid[i] = int(limit(kid[i],min_ep,max_ep))
        elif i == 2:
            kid[i] = int(limit(kid[i],min_bs,max_bs))
        else:
            kid[i] = int(limit(kid[i],min_node,max_node))
    return kid

def mutation(parent,kid,old_mark,new_mark):
    global mutation_strength
    p_target = 1.0/5
    if old_mark < new_mark:
        parent = kid
        ps = 1
    else:
        ps = 0
    for i in range(len(mutation_strength)):
        mutation_strength[i] = np.exp(1/4*(ps - p_target)/(1-p_target))*mutation_strength[i]





print('Start')
parent = [0,0,0,0,0]
kid = [0,0,0,0,0]
parent[0] = initialization(parent[0],max_lr,p_type=1)
parent[1] = initialization(parent[1],max_ep)
parent[2] = initialization(parent[2],max_bs)
parent[3] = initialization(parent[3],max_node)
parent[4] = initialization(parent[4],max_node)
'''
kid[0] = initialization(parent[0],max_lr,p_type=1)
kid[1] = initialization(parent[1],max_lr)
kid[2] = initialization(parent[2],max_lr)
kid[3] = initialization(parent[3],max_lr)
kid[4] = initialization(parent[4],max_lr)
'''
print(parent) #TODO

#read_data
#wilt dataset
#x_train,y_train,input_dim,output_dim = read_wilt_data('training.csv')
#x_test,y_test,input_dim,output_dim = read_wilt_data('testing.csv',1)
#balance dataset
x_train,y_train,x_test,y_test,input_dim,output_dim = read_balance_data('data.txt')

model = DBN_model(1,parent,input_dim,output_dim)
y_predict = make_kid(parent,x_train,y_train,x_test,model)

#For wilt dataset
#old_mark = wilt_fitness(y_predict,y_test) 
#For balance dataset
old_mark = balance_fitness(y_predict,y_test) 
for j in range(N_generation):
    kid = offspring(parent)
    model = DBN_model(1,kid,input_dim,output_dim)
    y_predict = make_kid(kid,x_train,y_train,x_test,model)

    #calculate the mark 
    #new_mark = wilt_fitness(y_predict,y_test)
    new_mark = balance_fitness(y_predict,y_test)

    #change mutation_strength
    mutation(parent,kid,old_mark,new_mark)

print('accuracy',old_mark)

    



