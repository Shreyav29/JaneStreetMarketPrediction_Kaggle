import os
import gc  #This is garbage collector 
import sys #System 
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
        
# from sklearn.preprocessing import StandardScaler as scale
# from sklearn.decomposition import PCA
# from sklearn.cluster import k_means


from random import seed
from random import randint
seed(1)


from matplotlib import pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna
from optuna.samplers import TPESampler
from sklearn.neural_network import MLPRegressor, MLPClassifier







def read_data():
    df = pd.read_csv("Inputs/train.csv")
    return df 

def convert_to_pickle(df, loc_with_name):
    with open(loc_with_name, 'wb') as fp:
        pickle.dump(df, fp)

def read_pickle(loc_with_name):
    with open (loc_with_name, 'rb') as fp:
        df = pickle.load(fp)
    return df 

def preprocess(df):
    #Dataset length 
    org_len = len(df)

    #Saving memory by changing the dtype
    for i in df:
        if df[i].dtype == np.float64:
            if (((df[i] < .0001) & (df[i] > -.0001)).mean()) < .001:
                df[i] = df[i].astype(np.float32)
                gc.collect()

    #Changing the data types 
    df.date = df.date.astype(np.int16)
    df.ts_id = df.ts_id.astype(np.int32)
    df.feature_0 = df.feature_0.astype(np.int32)

    #Sorting with respect to date
    df.sort_values(by = ['date','ts_id'],inplace = True)

    #Create a action column - 1 if the resp is >0 and 0 if resp < 0 
    df['action'] = np.where(df['resp'] > 0 , 1 , 0 )
    df.action = df.action.astype("category")
    
    return df


def return_per_day(resp, weight, action):
    returns = np.multiply(np.multiply(resp, weight), action)
    return sum(returns)

def sharpe_score(Pi_list):
    num = sum(Pi_list)
    den = np.sqrt(sum([i ** 2 for i in Pi_list]))
    sharpe = (num/den) * np.sqrt(252/len(Pi_list))
    utility_score = min(max(sharpe,0),6) * num
    
    return sharpe , utility_score

def generate_random_block(data, size):
    date_start = int(min(data.date))
    date_end = int(max(data.date)- size)
    
    block_start = randint(date_start, date_end)
    block_end = block_start + size 
    
    block_data = data.loc[data.date.isin([i for i in range(block_start, block_end +1 )])]
    
    return block_data
    
def generate_sequential_blocks(data, size):
    n_blocks = len(data)//size 
    date_start = int(min(data.date))
    date_end = int(max(data.date)- size)
    
    blocks_list = []
    
    for i in range(date_start, date_end):
        block_start = i 
        block_end = i+ size
        block_data = data.loc[data.date.isin([i for i in range(block_start, block_end +1 )])]
        
        blocks_list.append(block_data)
    return blocks_list


def split_train_test(sub_data, split):
    val = int(sub_data.date.nunique() * split)
    date_start = int(min(sub_data.date))
    date_end = int(max(sub_data.date))
    train = sub_data.loc[sub_data.date.isin([i for i in range(date_start, date_start + val +1 )])]
    test = sub_data.loc[sub_data.date.isin([i for i in range( date_start + val +1,date_end+1)])]
    
    return train , test 


def standardize(X_train, y_train, X_test, y_test, do_y):
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    mean_y = 0 
    std_y = 0 
    if do_y == True : 
        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mean_y)/std_y
        y_test = (y_test - mean_y)/std_y
    
    return X_train, y_train, X_test, y_test, mean_y , std_y

def inverse_standardize( y_pred,y_mean,y_std):
    
    for i in range(y_pred.shape[1]):
        y_pred[:,i]  = (y_pred[:,i] * y_std[i] + y_mean[i])
        
    return y_pred 

def create_action(threshold, y_pred, y_test, test_date, test_weight):
    #y_pred_df = pd.DataFrame(y_pred , columns = ['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp'], index = test_date)
    y_pred_df = pd.DataFrame(y_pred , columns = ['resp'], index = test_date)
    a = pd.DataFrame(test_weight, dtype = 'float')
    a.index = test_date
    y_test.index = test_date

    y_pred_df['product'] = y_pred_df['resp'] * a['weight']
    y_pred_df['weight'] = y_pred_df['product']/y_pred_df['resp']
    y_pred_df['action'] = np.where(y_pred_df['resp'] > threshold , 1 , 0 )
    print("Action counts are :")
    print(y_pred_df.action.value_counts())
    
    return y_pred_df, y_test 

def create_model(trial):
    max_depth = trial.suggest_int("max_depth", 2, 12)
    n_estimators = trial.suggest_int("n_estimators", 2, 600)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
    subsample = trial.suggest_uniform('subsample', 0.0001, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.0000001, 1)

    model = XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=666,
        tree_method='gpu_hist'
    )
    return model

def objective(trial):
    model = create_model(trial)
    model.fit(X_train, y_train)
    score = accuracy_score(
        y_train, 
        model.predict(X_train)
    )
    return score


def xgboost_helper(X_train, y_train, use_two_models=False):

    sampler = TPESampler(seed=666)

    params1 = {
        'max_depth': 8, 
        'n_estimators': 500, 
        'learning_rate': 0.01, 
        'subsample': 0.9, 
        'tree_method': 'gpu_hist',
        'random_state': 666,
        'n_jobs': 4,
        'verbosity':3
    }

    params3 = {
        'max_depth': 10, 
        'n_estimators': 500, 
        'learning_rate': 0.03, 
        'subsample': 0.9, 
        'colsample_bytree': 0.7,
        'tree_method': 'gpu_hist',
        'random_state': 666,
        'n_jobs': 4,
        'verbosity':3
    }

    print('model defined')
    model1 = XGBClassifier(**params1)
    print('model created')

    model1.fit(X_train, y_train, verbose='True')

    print("Built model 1")

    if use_two_models:
        model3 = XGBClassifier(**params3)
        model3.fit(X_train, y_train)

        return model1, model3
    
    else:
        return model1

    
def sigmoid(x, tau=1):
    return np.exp(x/tau)/(1+ np.exp(x/tau))