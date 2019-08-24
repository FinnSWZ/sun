import random
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


parents_number = 5
max_layer = 3
offspring_number = 10
N_generation = 20
old_best_offpsring = [0,0,0,0,0,0]
old_best_accuracy = 0
max_learning_rate = 0.8
max_epoch = 1000
max_batch_size = 256
max_node = 128


def generate_hp(parent_hyperparameter,hyperparameter):
    # selection operation
    select_list = range(1,parents_number)
    p1,p2 = random.sample(select_list,2)
    # corssover operation
    variation = parent_hyperparameter[0]
    for m in range(len(hyperparameter)):
        seq = [parent_hyperparameter[p1][m],parent_hyperparameter[p2][m]]
        variation[m] = random.sample(seq,1)[0]
    
    #mutation operation
    for n in range(len(hyperparameter)):
        if parent_hyperparameter[p1][n] < parent_hyperparameter[p2][n]:
            variation[n] = random.uniform(parent_hyperparameter[p1][n],parent_hyperparameter[p2][n])
        else:
            variation[n] = random.uniform(parent_hyperparameter[p2][n],parent_hyperparameter[p1][n])
        if n != 0:
            variation[n] = int(variation[n])
    return variation
