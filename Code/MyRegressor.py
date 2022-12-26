import cvxpy as cvx 
import numpy as np
import pandas as pd
from utils import *
from collections import deque
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * N
        self.alpha = alpha # Stores the alpha used for the L-1 penalty in regression
        self.feature_percentage = 20 # Used in finding the optimal split between features and samples in Task 1-5 and 2
        self.lr = 1e-2 # Setting the learning rate for Task 2
        self.correction_value = 5 # Used for weeding out x values that would corrupt the regressor 
        self.warm_start = 3 # 
        self.indices = 0 # Used for storing indices in Task 1-3
        self.indices_online_training = 0 # Used for storing indices generated in Task 2
        self.momentum = 0.1 # Used for training the regressor in Task 2
        self.annealing = 0.9 # Used for lowering the learning rate as the training continues in Task 2
    def select_features(self,trainX,trainY,percentage): # Input is the x,y and desired feature reduction
        return select_features_utils_func(trainX,trainY,self.alpha,percentage) # Call the ILP function to reduce features 
        
    def select_sample(self, trainX, trainY,percentage): # Input is  the x and y and desired sample reduction
        selected_trainX,selected_trainY = select_samples_utils_func(trainX,trainY,percentage) # Call the ILP function to reduce samples
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY,communication_cost): #Input is x, y and desired communication cost
        sample_cost = 100*(communication_cost/self.feature_percentage)
        (rows,_) = trainX.shape
        while(get_feature_sample_weights(self.feature_percentage,communication_cost,rows)): # Find the optimal split between features and sample reduction
            self.feature_percentage+=5
        sample_cost = 100*(communication_cost/self.feature_percentage) # Recalculate the sample percentage once the optimal percentage is found
        self.indices = select_features_utils_func(trainX,trainY,self.alpha,self.feature_percentage) # Store the indices needed to remove the features
        selected_trainX,selected_trainY = select_samples_utils_func(trainX[:,self.indices],trainY,sample_cost) # Pass the training data with features removed for sample reduction
        return selected_trainX,selected_trainY # Return the training dataset which is reduced in features and samples
    def train(self, trainX, trainY):  # input is x and y      
        (self.weight,self.bias) = train_linear_regressor_utils_func(trainX,trainY,self.alpha) # Call the LP function for finding the regressor
        (train_pred,train_error) = MyRegressor.static_eval(self.weight,self.bias,trainX,trainY) # Evaluate the regressor on the training dataset
        self.training_cost = trainX.shape[0]*trainX.shape[1] # Update the training cost
        return train_error
    def train_online(self, trainX, trainY,communication_cost,polyak=False): # Input is x,y, desired communication cost and whether to use Polyak Averaging
        self.correction_value
        sample_cost = 100*(communication_cost/self.feature_percentage)
        (rows,_) = trainX.shape
        while(get_feature_sample_weights(self.feature_percentage,communication_cost,rows)):# Find the optimal feature cost, sample cost split for a communication cost
            self.feature_percentage+=5
        sample_cost = (communication_cost/self.feature_percentage)
        communication_cost = communication_cost/100 
        features = trainX.shape[1]
        self.weight = np.zeros(features) # Initialise the weight and bias to zero. Performance is better than using randomized values
        self.bias = np.zeros(1)
        weight_gradient = np.zeros((features,1)) # Initialise the weight and bias gradients to zero as well
        bias_gradient = np.zeros((1))
        sent_data =0 # Use a tracker sent_data to collect how much is being sent to the central node
        collected_data = 0 # Use a tracker collected_data to collect how much data has come into the sensor
        historical_weights = np.zeros((features,1)) # Keep a track of the weights for Polyak averaging
        historical_bias = np.zeros((1)) # Keep a track of the bias for Polyak averaging
        indices_tracker = [] # Keep a track of the indices to remove for feature reduction
        for index in range(0,self.warm_start): # Warm start the regressor by sending over a couple of data points with all of the data
            y = trainY[index]
            (self.weight,self.bias,weight_gradient,bias_gradient) = online_training_loop_natural(trainX[index],y,self.weight,self.bias,self.alpha,learning_rate=self.lr,wg=weight_gradient,bg=bias_gradient,momentum=self.momentum,annealing=self.annealing)
            sent_data +=1*(self.feature_percentage/100)**(-1) # Since we're sending more than we want to adjust the sent and collected data as in 1 sample with 100% features is equal to 2 samples at 50% cost
            collected_data +=1*(self.feature_percentage/100)**(-1)
            indices_tracker.append(index) # Keep track of indices of data being sent
            historical_weights = np.hstack((historical_weights,self.weight.reshape(-1,1))) # Add to the historical weight for Polyak averaging
            historical_bias = np.hstack((historical_bias,self.bias.reshape(-1))) # Add to the historical bias for Polyak averaging
        self.indices_online_training = feature_correction_online_training(self.weight,self.feature_percentage) # Take the first few samples and extract the features by minimising the ILP used in Task 1-3
        self.weight = self.weight[self.indices_online_training] # Keep only the necessary weights, discards the weights for which the features were removed
        weight_gradient = weight_gradient[self.indices_online_training] # Keep only the necessary weight gradients
        historical_weights = historical_weights[self.indices_online_training,:] # Keep the necessary historical weights for Polyak averaging
        for index in range(self.warm_start,len(trainX)):
            y = trainY[index]
            x = trainX[index][self.indices_online_training] # Keep the x features we want
            (to_send) = pass_logic(sent_data,x,y,collected_data,sample_cost,self.correction_value,self.weight,self.bias) # Check if the data should be sent to the regressor
            if(to_send): # If it should be sent or not from the pass logic above
                sent_data+=1 # Increment the data 
                (self.weight,self.bias,weight_gradient,bias_gradient) = online_training_loop_natural(x,y,self.weight,self.bias,self.alpha,learning_rate=self.lr,wg=weight_gradient,bg=bias_gradient,momentum=self.momentum,annealing=self.annealing) # train the regressor
                historical_weights = np.hstack((historical_weights,self.weight.reshape(-1,1))) # Add to the historical weights
                historical_bias = np.hstack((historical_bias,self.bias.reshape(-1))) # Add to the historical weights
                indices_tracker.append(index) # Keep track of the indices of data being sent
            collected_data+=1; # Increment the number of datapoints the sensor has received
        self.training_cost = self.feature_percentage*(sent_data/collected_data)/100 # Compute the actual training cost of the regressor
        self.weight = self.weight.reshape(-1,1) # reshape the weights
        self.bias = self.bias.reshape(-1,1) # Reshape the bias
        if(polyak): # If Polyak averaging is true, take the average of all the weights and bias from the start
            historical_weights = np.sum(historical_weights,axis=1)/len(indices_tracker)
            historical_bias = np.sum(historical_bias)/len(indices_tracker)
            self.bias = historical_bias.reshape(-1,1)
            self.weight = historical_weights.reshape(-1,1)
        actual_training_set = trainX[indices_tracker,:] # Actual dataset the regressor was trained on, remove the samples it was not trained on
        actual_training_set = actual_training_set[:,self.indices_online_training] # Remove the features as well
        _,train_error = MyRegressor.static_eval(self.weight,self.bias,actual_training_set,trainY[indices_tracker]) # Extract training error 
        
        return self.training_cost, train_error
    def evaluate(self, X, Y): # If online training,use a different evaluate statement else go through the general route
        return MyRegressor.static_eval(self.weight,self.bias,X,Y) if (self.indices_online_training==0) else MyRegressor.static_eval(self.weight,self.bias,X[:,self.indices_online_training],Y)
    def get_params(self):
        return self.weight, self.bias # Return weight and bias
    
    @staticmethod
    def static_eval(weight,bias,X,Y): # Developed a static method to be able to call the function within the class
        Y = Y.reshape(-1,1)
        (rows,cols) = X.shape
        ones_vector = create_ones_vector(rows)
        pred = X@weight+ones_vector@bias
        error = mean_absolute_error(pred,Y)
        return pred,error
