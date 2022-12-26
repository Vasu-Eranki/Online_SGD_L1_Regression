import csv
import os
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from MyRegressor import *
from sklearn import preprocessing, model_selection,metrics
import pandas as pd
def prepare_data_gaussian():
    num_dtp=1000
    dims=500
    spars_param=0.8

    data = dict()
    feats = np.zeros((num_dtp, dims))
    for i in range(dims):
        dev = np.random.rand()
        x = np.random.uniform(-1,1,size=(num_dtp,)) + np.random.normal(0,dev,(num_dtp,))
        np.random.shuffle(x)
        feats[:,i] = x
    
    theta = 5 * np.random.rand(dims)
    ind = np.random.choice(np.arange(dims), round(dims*spars_param), replace=False)
    theta[ind] = np.random.random() * 0.0001

    bias = 0.3
    y = (feats @ theta).reshape(-1,1) + bias + np.random.normal(0,0.5,(num_dtp,1))

    # randomly split data into 60% train and 40% test set
    trainX, testX, trainYo, testYo = \
      model_selection.train_test_split(feats, y, 
      train_size=0.60, test_size=0.40, random_state=13)

    # normalize feature values
    scaler = preprocessing.StandardScaler()  
    data['trainX'] = scaler.fit_transform(trainX)  
    data['testX']  = scaler.transform(testX)
    
    # map targets to log-space
    data['trainY'] = trainYo.reshape(-1,)
    data['testY']  = testYo.reshape(-1,)

    return data

def prepare_data_news():
    # https://archive.ics.uci.edu/ml/datasets/online+news+popularity
    data = dict()
    
    filename = 'OnlineNewsPopularity/OnlineNewsPopularity.csv'
    
    # read the data
    allfeatnames = []
    textdata      = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(allfeatnames)==0:
                allfeatnames = row
            else:
                textdata.append(row)

    # put the data into a np array
    dataX = np.empty((len(textdata), len(allfeatnames)-3))
    dataY = np.empty(len(textdata))
    for i,row in enumerate(textdata):
        # extract features (remove the first 2 features and the last feature)
        dataX[i,:] = np.array([float(x) for x in row[2:-1]])
        # extract target (last entry)
        dataY[i] = float(row[-1])

    # extract feature names
    data['featnames'] = [x.strip() for x in allfeatnames[2:-1]]

    # extract a subset of data
    dataX = dataX[::6]
    dataY = dataY[::6]
    
    # randomly split data into 60% train and 40% test set
    trainX, testX, trainYo, testYo = \
      model_selection.train_test_split(dataX, dataY, 
      train_size=0.60, test_size=0.40, random_state=4487)

    # normalize feature values
    scaler = preprocessing.StandardScaler()  
    data['trainX'] = scaler.fit_transform(trainX)  
    data['testX']  = scaler.transform(testX)
    
    # map targets to log-space
    data['trainY'] = np.log10(trainYo)
    data['testY']  = np.log10(testYo)

    return data

def plot_result(result):
    ''' Input Format:
        task 1-2: result = {'taskID':'1-2', 'alpha':[], 'train_err':[], 'test_err':[]}
        task 1-3: result = {'taskID':'1-3', 'feat_num':[], 'train_err':[], 'test_err':[]}
        task 1-4: result = {'taskID':'1-4', 'sample_num':[], 'train_err':[], 'test_err':[]}
        task 1-5: result = {'taskID':'1-5', 'cost':[], 'train_err':[], 'test_err':[]}
        task 2: result = {'taskID':'2', 'cost':[], 'train_err':[], 'test_err':[]}

    '''
    if result['taskID'] == '1-2':
        x_value = result['alpha']
        x_label = 'penalty for sparsity'
        x_range = None
        x_scale = "log"
    
    elif result['taskID'] == '1-3':
        x_value = result['feat_num']
        x_label = 'number of features'
        x_range = (0,1)
        x_scale = "linear"
    
    elif result['taskID'] == '1-4':
        x_value = result['sample_num']
        x_label = 'number of samples'
        x_range = (0,1)
        x_scale = "linear"

    else:  # result['taskID'] == '1-5' or '2'
        x_value = result['cost']
        x_label = 'communication cost during training'
        x_range = (0,1)
        x_scale = "linear"
        
    plt.plot(x_value, result['train_err'], label = 'train_error', marker='x', markersize=8)
    plt.plot(x_value, result['test_err'], label = 'test_error', marker='o', markersize=8)

    plt.xlabel(x_label, fontsize=12) 
    plt.ylabel('MAE', fontsize=12)
    plt.title("Result of Task " + result['taskID'], fontsize=14)
    plt.legend()
    plt.xscale(x_scale)
    plt.xlim(x_range)
    plt.show()
    return 
def create_ones_vector(shape):
    return np.ones((shape,1)) # Create a column vector of ones for matching the shape of the bias during regression

def create_dictionary(task,second_var,second_var_results,training_error,testing_error): # Created a function to generate the desired dictionary for plotting results
    return {'taskID':task,second_var:second_var_results,'train_err':training_error,'test_err':testing_error}
    
def collect_train_test_results(x_train,y_train,x_test,y_test,regressor): # Input is the entire dataset with the regressor class to collect the training and testing error
    training_error =regressor.train(x_train,y_train)
    testing_predictions,test_error = regressor.evaluate(x_test,y_test)
    return training_error,test_error
    
def select_features_utils_func(x_train,y_train,alpha,feat_percentage): # Input is x and y and desired feature reduction
    (rows,cols) = x_train.shape    
    y_train = y_train.reshape(-1,1)
    weight = cvx.Variable((cols,1))
    bias = cvx.Variable((1,1))
    ones_vector = create_ones_vector(rows)
    objective = cvx.Minimize(cvx.norm(y_train-x_train@weight-ones_vector@bias,1)/rows+alpha*cvx.norm(weight,1))# Creates a regressor between x and y
    prob = cvx.Problem(objective)
    prob.solve(verbose=False)
    actual_weight = weight.value
    actual_bias = bias.value
    del prob,objective,bias,actual_bias,ones_vector,weight,y_train # Removes dummy variables
    mask = cvx.Variable((cols,cols),diag=True) # Create a mask to find the optimal features to mask, set it to be a diagonal matrix
    mask_dummy = cvx.Variable((cols,cols),boolean=True) # Create a dummy variable and make it boolean. 
    objective = cvx.Minimize(cvx.norm(actual_weight-mask@actual_weight,1))
    constraints= [cvx.trace(mask)==round(feat_percentage*cols/100),cvx.sum(mask)==round(feat_percentage*cols/100),mask==mask_dummy] # Trace(Mask)=k, Sum(Mask)=k, the mask needs to be diagonal and boolean hence the last constraint
    problem = cvx.Problem(objective,constraints)
    problem.solve(verbose=False,solver="GLPK_MI") # Use a Mixted Integer Programming Solver to solve ILP
    select_feat = [i for i in range(0,cols) if round(mask.value.toarray()[i,i],1)>0] # Find the features to be kept by inspecting the mask diagonal where element = 1
    return select_feat # Return features to keep

def select_samples_utils_func(x_train,y_train,sample_percentage):# Input is x, y and the desired sample reduction 
    (rows,cols) = x_train.shape
    samples_mask = cvx.Variable((rows,rows),diag=True) # Creata mask that is diagonal
    samples_mask_dummy = cvx.Variable((rows,rows),boolean=True) #Create a mask that is boolean. Both variables are made equal,making it diagonal and boolean
    constraints = [cvx.trace(samples_mask)==round(sample_percentage*rows/100),cvx.sum(samples_mask)==round(sample_percentage*rows/100),samples_mask==samples_mask_dummy]# Trace(mask)=k, sum(mask)=k, mask should be both diagonal and boolean
    objective  = cvx.Minimize(cvx.norm(y_train-samples_mask@y_train,1))
    problem = cvx.Problem(objective,constraints)
    problem.solve(verbose=False,solver="GLPK_MI") # Use a Mixed Integer Program to solve this 
    select_indexes = [i for i in range(0,rows) if round(samples_mask.value.toarray()[i,i],1)>0] # Extract the indices  of the samples to keep
    selected_trainX = x_train[select_indexes,:] #Filter out the unwanted samples in X
    selected_trainY = y_train[select_indexes] # Filter out the unwanted samples in Y
    return selected_trainX,selected_trainY # Return the new training dataset
def train_linear_regressor_utils_func(x_train,y_train,alpha): # Input is x, y and the regularization 
    (rows,cols) = x_train.shape
    y_train = y_train.reshape(-1,1)
    weight = cvx.Variable((cols,1))
    bias = cvx.Variable((1,1))
    ones_vector = create_ones_vector(rows) # Create a one vector to match the shape of the bias with x and y
    objective = cvx.Minimize(cvx.norm(y_train-x_train@weight-ones_vector@bias,1)/rows+alpha*cvx.norm(weight,1)) #Use the original primal problem without reformulation
    prob = cvx.Problem(objective)
    prob.solve(verbose=False)
    actual_weight = weight.value # Extract the values of the weight
    actual_bias = bias.value # Extract the values of the bias
    return (actual_weight,actual_bias) # Return the values


def get_feature_sample_weights(feature_cost,communication_cost,rows): # Takes the feature cost, communication cost and the total # of samples
    sample_cost = 100*(communication_cost/feature_cost)
    if(sample_cost*rows)<1:
        return True
    return False if (sample_cost<=100) else True # If the sample cost is >1, it is not a feasible split. 

def online_training_loop_natural(x_train,y_train,weight,bias,alpha,learning_rate,wg,bg,momentum,annealing):# This is for Task 2, it takes x,y the weight, bias, the regularization term, the previous weight gradients, bias gradient, momentum and an annealing factor
    x_train = x_train.reshape(-1,1)
    for i in range(0,5): # Train the regressor 5 times on the same data
        error = weight.T@x_train+bias-y_train # Calculate the error 
        sign_value = np.sign(error) # Find the sign of the error
        for i in range(0,len(weight)): # Iterate over the weights
            weight[i] = weight[i]-learning_rate*(sign_value*x_train[i]+alpha*np.sign(weight[i]) +momentum*wg[i]) # Update the weight using Steepest Gradient Descent
            wg[i] = sign_value*x_train[i]+alpha*np.sign(weight[i]) # Calculate the new weight gradients
        bias  = bias-learning_rate*(sign_value+momentum*bg) # Update the bias
        bg = sign_value # Update the bias gradient
        learning_rate = learning_rate*annealing# Decrease the learning rate as the regressor is being trained on the datapoint
   
    return weight,bias,wg,bg# return the weight, bias, and the gradients of the weight and bias (Downlink communication is only the weight and bias from central node to sensor. 
    
def pass_logic(history,x_data,y,total_data,communication_cost,correction_value,weight,bias):# Used to figure out if to send or not send the data
    if(total_data==0): # If its the first data, do not apply logic
        return True
    elif(history/total_data <=communication_cost):# If abs normalised version of error > correction value, it will corrupt the regressor do not send
        pred = weight.T@x_data+bias
        error = abs(y-pred)/y
        if(error<=correction_value):
            return True
    return False

def feature_correction_online_training(weights,feat_percentage): #Run an ILP to extract which features to remove 
    cols = len(weights)
    mask = cvx.Variable((cols,cols),diag=True)
    mask_dummy = cvx.Variable((cols,cols),boolean=True)
    objective = cvx.Minimize(cvx.norm(weights-mask@weights,1))
    constraints= [cvx.trace(mask)==int(feat_percentage*cols/100),cvx.sum(mask)==int(feat_percentage*cols/100),mask==mask_dummy]
    problem = cvx.Problem(objective,constraints)
    problem.solve(verbose=False,solver="GLPK_MI")
    select_feat = [i for i in range(0,cols) if round(mask.value.toarray()[i,i],1)>0]
    return select_feat
    