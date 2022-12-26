import numpy as np
from MyRegressor import *
from utils import *
from tqdm import tqdm
def main():
    data = prepare_data_gaussian()
    #task_1_2(data)
    #task_1_3(data,3e-2)
    #task_1_4(data,3e-2)
    #task_1_5(data,3e-2)
    
    task_2(data,3e-2,False)
    data_news = prepare_data_news()
    #task_1_2(data_news)
    #task_1_3(data_news,1e-2)
    #task_1_4(data_news,1e-2)
    #task_1_5(data_news,1e-2)
    task_2(data_news,1e-2,True)
def task_1_2(data): #Task 1_2 varying the alpha
    alpha = [0,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4,1e-3,2e-3,3e-3,4e-3,5e-3,1e-2,2e-2,3e-2,4e-2,5e-2,1e-1,2e-1,3e-1,4e-1,5e-1]
    train_error = []
    testing_error = []
    for i in (range(0,len(alpha))):
        regressor = MyRegressor(alpha[i])
        train_err,test_err = collect_train_test_results(data['trainX'],data['trainY'],data['testX'],data['testY'],regressor)
        train_error.append(train_err)
        testing_error.append(test_err)
    results_dictionary = create_dictionary('1-2','alpha',alpha,train_error,testing_error)
    plot_result(results_dictionary)    
def task_1_3(data,alpha): # Task 1-3 varying the feature size
    percentage = [1,5,10,20,25,30,40,50,60,70,80,90,100]
    n_features = [i/100 for i in percentage]
    regressor = MyRegressor(alpha)
    train_error = []
    testing_error = []
    for i in (range(0,len(percentage))): 
        indices = regressor.select_features(data['trainX'],data['trainY'],percentage[i])
        train_err,test_err = collect_train_test_results(data['trainX'][:,indices],data['trainY'],data['testX'][:,indices],data['testY'],regressor)
        train_error.append(train_err)
        testing_error.append(test_err)
    results_dictionary = create_dictionary('1-3','feat_num',n_features,train_error,testing_error)
    plot_result(results_dictionary)
    return
def task_1_4(data,alpha): #Task 1-4, varying the sample size
    percentage = [1,5,10,20,25,30,40,50,60,70,80,90,100]
    n_samples = [i/100 for i in percentage]
    regressor = MyRegressor(alpha)
    train_error = []
    testing_error  = []
    for i in (range(0,len(percentage))):
        x_train,y_train = regressor.select_sample(data['trainX'],data['trainY'],percentage[i])
        train_err,test_err  = collect_train_test_results(x_train,y_train,data['testX'],data['testY'],regressor)
        train_error.append(train_err)
        testing_error.append(test_err)
    results_dictionary = create_dictionary('1-4','sample_num',n_samples,train_error,testing_error)
    plot_result(results_dictionary)
    return
def task_1_5(data,alpha):# Task 1-5 varying both the sample size and features
    sample_percentage = [1,5,10,20,25,30,40,50,60,70,80,90,100]
    regressor = MyRegressor(alpha)
    train_error = []
    testing_error = []
    for i in (range(0,len(sample_percentage))):
        x_train,y_train = regressor.select_data(data['trainX'],data['trainY'],sample_percentage[i])
        indices = regressor.indices
        train_err,test_err = collect_train_test_results(x_train,y_train,data['testX'][:,indices],data['testY'],regressor)
        train_error.append(train_err)
        testing_error.append(test_err)
    sample_percentage = [i/100 for i in sample_percentage]
    results_dictionary = create_dictionary('1-5','cost',sample_percentage,train_error,testing_error)
    plot_result(results_dictionary)
def task_2(data,alpha,polyak): # Online Regression
    communication_cost = [1,5,10,20,25,30,40,50,60,70,80,90,100]
    regressor = MyRegressor(alpha)
    train_error = []
    testing_error = []
    for i in (range(0,len(communication_cost))):
        (training_cost,train_err) = regressor.train_online(data['trainX'],data['trainY'],communication_cost[i],polyak)
        _,test_err = regressor.evaluate(data['testX'],data['testY'])
        train_error.append(train_err)
        testing_error.append(test_err)
    communication_cost = [i/100 for i in communication_cost]
    results_dictionary = create_dictionary('2','cost',communication_cost,train_error,testing_error)
    plot_result(results_dictionary)
if __name__=="__main__":
    main()