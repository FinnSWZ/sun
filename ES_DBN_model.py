from DNN_model import DBN_model,make_kid
from read_wilt_data import read_wilt_data, wilt_fitness
from read_Balance_data import read_balance_data, balance_fitness
import time
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
import os 
import math 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

max_layer = 3       # the number of the upper bound of hidden layer
N_generation = 5   # iteration time
parents_number = 5
offspring_number =10   
max_learning_rate = 0.8
max_epoch = 2000          
max_batch_size = 256
max_node = 128
acc = []
old_best_accuracy = 0
p = 0
new_variation = [0,0,0,0,0,0]
p_lr = 0
p_ep = 0
p_bs = 0
p_n = 0

## generate offspring
def generate_hp(parent_hyperparameter,offspring_seq,generation=1):
    '''
    #first method 
    #crossover
    #choose two parents randomly
    select_list = range(0,parents_number)
    p1,p2 = random.sample(select_list,2)
    
    variation = parent_hyperparameter[0]
    for m in range(len(hyperparameter)):
        seq =   [parent_hyperparameter[p1][m],parent_hyperparameter[p2][m]]
        variation[m] = random.sample(seq,1)[0]
    
    # mutation  

   
    variation[0] += 1/(offspring_seq + 1) * mutation_strength_lr * np.random.randn()
    variation[1] += int(1/(offspring_seq + 1) * mutation_strength_ep * np.random.randn())
    variation[2] += int(1/(offspring_seq + 1) * mutation_strength_bs * np.random.randn())
    variation[3] += int(1/(offspring_seq + 1) * mutation_strength_n * np.random.randn())
    variation[4] += int(1/(offspring_seq + 1) * mutation_strength_n * np.random.randn())
    variation[5] += int(1/(offspring_seq + 1) * mutation_strength_n * np.random.randn())

    #limit the range of parameters
    variation = hy_limit(variation)

    return variation
    '''
    
    # second mutation method(CMA_ES)
    # crossover operation
    x_0 = 0
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x_5 = 0
    
    for i in range(len(parent_hyperparameter)):
        global new_variation
        x_0 += parent_hyperparameter[i][0]
        x_1 += parent_hyperparameter[i][1]
        x_2 += parent_hyperparameter[i][2]
        x_3 += parent_hyperparameter[i][3]
        x_4 += parent_hyperparameter[i][4]
        x_5 += parent_hyperparameter[i][5]
    old_variation = new_variation
    new_variation[0] = x_0/len(parent_hyperparameter)
    new_variation[1] = x_1/len(parent_hyperparameter)
    new_variation[2] = x_2/len(parent_hyperparameter)
    new_variation[3] = x_3/len(parent_hyperparameter)
    new_variation[4] = x_4/len(parent_hyperparameter)
    new_variation[5] = x_5/len(parent_hyperparameter)
    #mutation operation
    # CSA formula
    global mutation_strength_lr,mutation_strength_bs,mutation_strength_ep,mutation_strength_n,p_lr,p_ep,p_bs,p_n
    n = 4
    c = 10/(n+20) 
    d = max(1,(3*parents_number)/(n+10)) + c
    e = math.sqrt(n) * (1 - 1/(4 * n) + 1/(21*n*n))
    if generation == 0:
        p_lr = (1-c) * p_lr + math.sqrt(c*(2-c) * math.sqrt(parents_number)/mutation_strength_lr * (new_variation[0]-old_variation[0]))
        p_ep = (1-c) * p_ep + math.sqrt(c*(2-c) * math.sqrt(parents_number)/mutation_strength_ep * (new_variation[1]-old_variation[1]))
        p_bs = (1-c) * p_bs + math.sqrt(c*(2-c) * math.sqrt(parents_number)/mutation_strength_bs * (new_variation[2]-old_variation[2]))
        p_n = (1-c) * p_n + math.sqrt(c*(2-c) * math.sqrt(parents_number)/mutation_strength_n * (new_variation[3]-old_variation[3]))
        mutation_strength_lr = mutation_strength_lr * math.exp((c/d) * abs(p_lr)/e - 1)
        mutation_strength_ep = mutation_strength_ep * math.exp((c/d) * abs(p_ep)/e - 1)
        mutation_strength_bs = mutation_strength_bs * math.exp((c/d) * abs(p_bs)/e - 1)
        mutation_strength_n = mutation_strength_n * math.exp((c/d) * abs(p_n)/e - 1)
    





    new_variation[0] = new_variation[0] + mutation_strength_lr * np.random.randn()
    new_variation[1] = int(new_variation[1] + mutation_strength_ep * np.random.randn() )
    new_variation[2] = int(new_variation[2] + mutation_strength_bs * np.random.randn())
    new_variation[3] = int(new_variation[3] + mutation_strength_n * np.random.randn())
    new_variation[4] = int(new_variation[4] + mutation_strength_n * np.random.randn())
    new_variation[5] = int(new_variation[5] + mutation_strength_n * np.random.randn())
    #print(variation)



    #limit the range of parameters
    new_variation = hy_limit(new_variation)
    
    return new_variation

# limit the range of parameters  
def hy_limit(hyperparameter):
    hyperparameter[0] = max(hyperparameter[0],0.01)
    hyperparameter[0] = min(hyperparameter[0],0.8)
    hyperparameter[1] = max(hyperparameter[1],32)
    hyperparameter[1] = min(hyperparameter[1],2000)  
    hyperparameter[2] = max(hyperparameter[2],8)
    hyperparameter[2] = min(hyperparameter[2],256)
    hyperparameter[3] = max(hyperparameter[3],1)
    hyperparameter[3] = min(hyperparameter[3],32)
    hyperparameter[4] = max(hyperparameter[4],1)
    hyperparameter[4] = min(hyperparameter[4],32)
    hyperparameter[5] = max(hyperparameter[5],1)
    hyperparameter[5] = min(hyperparameter[5],32)
    return hyperparameter

print('Start!')
begin_time = time.asctime(time.localtime(time.time()))

#initialize the parents attribution
parent_hyperparameter = []
for i in range(parents_number):
    hyperparameter = []
    hyperparameter_0 = np.random.rand() * max_learning_rate
    hyperparameter_1 = int(np.random.rand() * max_epoch + 500)
    hyperparameter_2 = int(np.random.rand() * max_batch_size)
    hyperparameter_3 = int(np.random.rand() * max_node)
    hyperparameter_4 = int(np.random.rand() * max_node)
    hyperparameter_5 = int(np.random.rand() * max_node)
    hyperparameter.append(hyperparameter_0)
    hyperparameter.append(hyperparameter_1)
    hyperparameter.append(hyperparameter_2)
    hyperparameter.append(hyperparameter_3)
    hyperparameter.append(hyperparameter_4)
    hyperparameter.append(hyperparameter_5)
    parent_hyperparameter.append(hyperparameter)

#read_data
#wilt dataset
#x_train,y_train,input_dim,output_dim = read_wilt_data('training.csv')
#x_test,y_test,input_dim,output_dim = read_wilt_data('testing.csv',1)

#balance dataset
x_train,y_train,x_test,y_test,input_dim,output_dim = read_balance_data('data.txt')
print('1')

#increase one layer for each iteration
for i in range(max_layer):
    mutation_strength_lr = 3    
    mutation_strength_ep = 100  
    mutation_strength_bs = 32  
    mutation_strength_n = 32  
    for j in range(N_generation): 
        
        dictionary = {}     #use to choose the best offspring
        new_list = []       # use to save the new_hyperparameter

        #generate the offspring
        for a in range(offspring_number):
            # get new hyperparameter and node number
            hyperparameter = generate_hp(parent_hyperparameter,j,a)


            ''' constructe the neural network with new node '''
            #1 DBN model 
            model = DBN_model(i,hyperparameter,input_dim,output_dim)
                        
            #get prediction 
            y_predict = make_kid(hyperparameter,x_train,y_train,x_test,model)

            #calculate the accuracy 
            #accuracy = wilt_fitness(y_predict,y_test)
            accuracy = balance_fitness(y_predict,y_test)

            # add the newparameter into an array            
            new_list.append([])
            for x in hyperparameter:
                new_list[a].append(x)

            # add the newparameter into an array

            #constructe a dictionary 
            #value is iteration time, key is performance of offspring
            dictionary[a] = accuracy


        # sorting according to the accuracy
        order = sorted(dictionary.items(),key=lambda item:item[1],reverse=True)

        
        #initialize parents
        parent_hyperparameter = []

        
        for n in range(parents_number):

            # choose the new parents for next generation
            new_hyperparameter = new_list[order[n][0]]
            parent_hyperparameter.append(new_hyperparameter)

        # choose the best offspring for the last generation
        new_best_offspring = new_list[order[0][0]]
        new_best_accuracy = order[0][1]
        print('accuracy:',new_best_accuracy)
        # compare the performance between this offspring and other module
        # best hyperparameter and topology
        if old_best_accuracy < new_best_accuracy:
            old_best_offspring = new_best_offspring
            old_best_accuracy = new_best_accuracy
            best_layer = i 
            if j == 5:
                print('1')
            acc.append(old_best_accuracy)    

        

#delete extra nodes
for i in range(len(old_best_offspring) - best_layer - 4):
    old_best_offspring[i+best_layer+4] = 0
finish_time = time.asctime(time.localtime(time.time()))




print('acc:',acc)
print('finial best accuracy:',old_best_accuracy)
print('finial best offspring',old_best_offspring)
print('best_layer:',best_layer)
print('begin_time',begin_time)
print('finish_time',finish_time)
