from GA import generate_hp
from DNN_model import CNN_model
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras import optimizers
from keras import initializers
from keras.utils import to_categorical
import random
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

max_layer = 2       # the number of the upper bound of hidden layer
N_generation = 5   # iteration time
parents_number = 6
offspring_number = 6
max_learning_rate = 0.8
max_epoch = 1000          
max_batch_size = 256
max_node = 128
acc = []
old_best_accuracy = 0

# limit the range of parameters  
def hy_limit(hyperparameter):
    hyperparameter[0] = max(hyperparameter[0],0.01)
    hyperparameter[0] = min(hyperparameter[0],0.8)
    hyperparameter[1] = max(hyperparameter[1],32)
    hyperparameter[1] = min(hyperparameter[1],1000)  #TODO for test
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

#initialize the parents attribution
parent_hyperparameter = []
for i in range(parents_number):
    hyperparameter = []
    hyperparameter_0 = np.random.rand() * max_learning_rate
    hyperparameter_1 = int(np.random.rand() * max_epoch)
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
x_train,y_train = mnist.load_data()[0]
x_train = x_train.reshape(-1,28,28,1)
x_train = x_train.astype('float64')
x_train = x_train[0:10000]
x_train /=255
y_train =y_train[0:10000]
y_train = to_categorical(y_train,10)
x_test,y_test =mnist.load_data()[1]
x_test = x_test.reshape(-1,28,28,1)
x_test = x_test.astype('float64')
x_test = x_test[0:100]
x_test /= 255
y_test = y_test[0:100]
y_test = to_categorical(y_test,10)
#increase one layer for each iteration
for i in range(max_layer):
    for j in range(N_generation): 
        
        dictionary = {}     #use to choose the best offspring
        new_list = []       # use to save the new_hyperparameter

        #generate the offspring
        for a in range(offspring_number):
            # get new hyperparameter and node number
            hyperparameter = generate_hp(parent_hyperparameter,hyperparameter)
            print('i:',i,' and hy:',hyperparameter)


            # construct CNN model and get the fitness
            performance = CNN_model(i,hyperparameter,x_train,y_train,x_test,y_test)
            accuracy = performance[1]
                        
            #calculate the fitness
            #accuracy = make_kid_CNN(hyperparameter,x_train,y_train,x_test,y_test,model)

            # add the newparameter into an array
            new_list.append([])
            for x in hyperparameter:
                new_list[a].append(x)
            '''selection operation'''
            #constructe a dictionary 
            #value is iteration time, key is performance of offspring
            dictionary[a] = accuracy
            print(i,'hidden layers')   # for test TODO
            print(j,'th offspring')     # for test TODO
            print('the',a,'th number')  # for test TODO

        # sorting according to the accuracy
        order = sorted(dictionary.items(),key=lambda item:item[1],reverse=True)

        
        #initialize parents
        
        if j == 0:
            old_best_offspring = parent_hyperparameter[0]
        else:
            old_best_offspring = new_best_offspring
        parent_hyperparameter = []
        
        for n in range(parents_number):

            # choose the new parents for next generation 
            new_hyperparameter = new_list[order[n][0]]
            parent_hyperparameter.append(new_hyperparameter)
            #use the best parent replace the worst offspring
        parent_hyperparameter[n] = old_best_offspring
        # choose the best offspring for the last generation
        new_best_offspring = new_list[order[0][0]]
        new_best_accuracy = order[0][1]
        print('accuracy:',new_best_accuracy)

        
        # compare the performance between this offspring and other module
  
        if old_best_accuracy < new_best_accuracy:
            old_best_offspring = new_best_offspring
            old_best_accuracy = new_best_accuracy
            best_layer = i 
            acc.append(old_best_accuracy)           #TODO for test
        if i == 1:
            print('1')

        

#delete extra nodes
for i in range(len(old_best_offspring) - best_layer - 4):
    old_best_offspring[i+best_layer+4] = 0


print('acc:',acc)
print('finial best accuracy:',old_best_accuracy)
print('finial best offspring',old_best_offspring)
print('best_layer:',best_layer)
