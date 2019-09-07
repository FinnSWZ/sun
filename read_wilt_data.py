import numpy as np
import pandas as pd
import operator

# read wilt dataset
def read_wilt_data(file,purpose = 0):
    
    total_data = np.loadtxt(file,delimiter=',',dtype=str)

    x_train = []
    y_train = []

    #data processing
    for i in range(len(total_data)-1):
        x = total_data[i+1]
        column_label = x[0]
        input_initial = []

        #encoding the label 
        if column_label == 'w':
            column_label = [1,0]
        else:
            column_label = [0,1]
        column_2 = int(float(x[1]))
        column_3 = int(float(x[2]))
        column_4 = int(float(x[3]))
        column_5 = int(float(x[4]))
        column_6 = int(float(x[5]))
        #normalization
        
        column_2 = (column_2 - 0)/(36-0)
        column_3 = (column_3 - 164)/(180-164)
        column_4 = (column_4 - 59)/(150-59)
        column_5 = (column_5 - 86)/(200-86)
        column_6 = (column_6 - 0)/(43-0)
        

        input_initial = [column_2,column_3,column_4,column_5,column_6]
        y_train.append(column_label)
        x_train.append(input_initial)
    x_train = np.mat(x_train)
    if purpose == 0:
        y_train = np.mat(y_train)


    input_dim = 5
    output_dim = 2
    return x_train,y_train,input_dim,output_dim


# Decode the prediction of wilt dataset
def wilt_fitness(y_predict,y_test):
    x = 0
    n = 0
    for j in range(len(y_predict)):
        if y_predict[j][0] < 0.5:
            n += 1
        if  y_predict[j][0] > y_predict[j][1]:
            str_predict = [1,0]
        else:
            str_predict = [0,1]
        if np.all(str_predict == y_test[j]):
            x = x + 1 
    accuracy = x/len(y_predict)
    print(n)

    return accuracy



