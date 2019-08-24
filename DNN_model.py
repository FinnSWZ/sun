import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import initializers
from keras import optimizers
from keras.optimizers import Adadelta

## construt DBN model
def DBN_model(time,hyperparameter,input_dim,output_dim):
    model = Sequential()
    model.add(Dense(hyperparameter[3],kernel_initializer='random_uniform',input_dim=input_dim))
    model.add(Activation('relu'))  
    print('one layer')
    if time > 0:
        model.add(Dense(hyperparameter[4],kernel_initializer='random_uniform'))
        model.add(Activation('relu')) 
        print('two layers')
        if time > 1:
            model.add(Dense(hyperparameter[5],kernel_initializer='random_uniform'))
            model.add(Activation('relu')) 
            print('three layers')
    else:
        pass
    model.add(Dense(output_dim,kernel_initializer='random_uniform'))
    model.add(Activation('sigmoid'))
    
    return model


#construct CNN model

def CNN_model(time,hyperparameter,x_train,y_train,x_test,y_test,input_shape=[28,28,1],output_dim=10):
    
    model = Sequential()
    model.add(Conv2D(filters = hyperparameter[3],kernel_size=(5,5),padding='Same', activation='relu',input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    if time > 0:
        model.add(Conv2D(filters = hyperparameter[4],kernel_size=(5,5),padding='Same',activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        print(2)
        if time > 1:
            model.add(Conv2D(filters = hyperparameter[4],kernel_size=(5,5),padding='Same',activation='relu'))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            print(3)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(output_dim,activation='softmax'))
    #adam = optimizers.Adam(lr=hyperparameter[0],beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    adadelta = keras.optimizers.Adadelta(lr=hyperparameter[0],rho=0.95)
    model.compile(loss='mse',optimizer=adadelta,metrics=['accuracy'])
    model.fit(x_train, y_train,batch_size=hyperparameter[2],epochs=hyperparameter[1])
    accuracy = model.evaluate(x_test,y_test,verbose=1)
    return accuracy
    
def make_kid(hyperparameter,x_train,y_train,x_test,model):
 
    # compile the model with new hyperparameter
    #adam = optimizers.Adam(lr=hyperparameter[0],beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    adadelta = keras.optimizers.Adadelta(lr = 0.7,rho = 0.95)
    model.compile(loss='mse',optimizer=adadelta)

    model.fit(x_train,y_train,epochs=500,batch_size=hyperparameter[2])
    y_predict = model.predict(x_test)
    return y_predict

    
        