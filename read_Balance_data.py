import numpy as np
from sklearn.model_selection import train_test_split


def read_balance_data(file):
    total_data = np.loadtxt(file,dtype=str)
    x = total_data[0]

    label = []
    input = []
    label_s = []
    

    #data processing
    for i in range(len(total_data)):
    #for i in range(5):
        input_initial = []
        x = total_data[i]
        column_label = x[0]

        #encoding the label data  

        label_s.append(column_label)
        if column_label == 'B':
            column_label = [1,0,0]
        elif column_label == 'R':
            column_label = [0,1,0]
        else:
            column_label = [0,0,1]
        
        column_2 = float(x[2])
        column_3 = float(x[4])
        column_4 = float(x[6])
        column_5 = float(x[8])

        input_initial = [column_2,column_3,column_4,column_5]
    
        label.append(column_label)
        input.append(input_initial)



    x_train,x_test,y_train,y_test = train_test_split(input,label,test_size=0.3)
    x_train = np.mat(x_train)
    x_test = np.mat(x_test)
    y_train = np.mat(y_train)
    input_dim = 4
    output_dim = 3

    return x_train,y_train,x_test,y_test,input_dim,output_dim  


#fitness function
def balance_fitness(y_predict,y_test):
    x = 0
    for j in range(len(y_predict)):
        if y_predict[j][0] > y_predict[j][1] and y_predict[j][0] > y_predict[j][2]:
            str_predict = [1,0,0]
        elif y_predict[j][1] > y_predict[j][0] and y_predict[j][1] > y_predict[j][2]:
            str_predict = [0,1,0]
        else:
            str_predict = [0,0,1]
        if str_predict == y_test[j]:
            x = x + 1 
    accuracy = x/len(y_predict)

    return accuracy

