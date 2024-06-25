#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates embeddings and hyperparameter results for transE and holE

input: data/triples.csv

output: 

    embeddings_final/transE/transE_50_5_50_0.1_0.1 (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
        - entities.csv
        - entity_embeddings.csv
        - relations.csv
        - relation_embeddings.csv
        - train_triples.csv

    output/transE_holE_hyperparam_result.csv

Define below mentioned lists for training multiple models with different parameter values
lrs=[0.1, 0.01, 0.05]                   (learning rate)
embed_size=[40, 50]                     (embedding size)
epochs=[100, 500, 1000]                 (epochs)
model_name=['transE', 'holE']           (model name)
"""

# Import the package modules from the "eduTransE_HolE" directory.
# This directory contain the module files that we have updated to support
# functionality where we can get the list of losses for each epoch.
import numpy as np
from eduTransE_HolE.latent_features import TransE, HolE
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from eduTransE_HolE.evaluation import evaluate_performance, mrr_score, mr_score, hits_at_n_score
from eduTransE_HolE.utils import create_tensorboard_visualizations
import os
from datetime import datetime


# Load the input data which contains the triples with the weights
X = pd.read_csv("data/triples.csv")

# Below commented code remove the "topics, concept_vocab_index, concept_vocab" triples
# Uncomment it if you want to remove those relations
"""
X.rename(columns = {'prob':'weights'}, inplace = True)
X['variable'].unique()
index = X[ (X['head'].astype(str).str.match("topic+"))].index
X.drop(index, inplace=True)
"""

# Create sub-dataframes for each type of relations
l_text_topics = X[X['variable']=='l_text_topics']
concept_vocab_index = X[X['variable']=='concept_vocab_index']
prerequisite = X[X['variable']=='prerequisite']
level = X[X['variable']=='level']

# Shuffle the sub-dataframes
l_text_topics = l_text_topics.sample(frac=1)
concept_vocab_index = concept_vocab_index.sample(frac=1)    
prerequisite = prerequisite.sample(frac=1)
level = level.sample(frac=1)


# Below commented code can be used to scale the weight values of the triples

# scaler = MinMaxScaler()
# l_text_topics['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(l_text_topics['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# concept_vocab_index['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(concept_vocab_index['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# prerequisite['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(prerequisite['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# level['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(level['weights'])), columns = ['weights'])


# Split the sub-dataframe of each relation into train, val, and test set with the ratio of 70:15:15
fractions = np.array([0.7, 0.15, 0.15])

ltt_train, ltt_val, ltt_test = np.array_split(l_text_topics, (fractions[:-1].cumsum() * len(l_text_topics)).astype(int))

cvi_train, cvi_val, cvi_test = np.array_split(concept_vocab_index, (fractions[:-1].cumsum() * len(concept_vocab_index)).astype(int))

pre_train, pre_val, pre_test = np.array_split(prerequisite, (fractions[:-1].cumsum() * len(prerequisite)).astype(int))

lvl_train, lvl_val, lvl_test = np.array_split(level, (fractions[:-1].cumsum() * len(level)).astype(int))

# Merge the sub-dataframe for each relation into train, val, test dataframes
train = pd.concat([ltt_train, cvi_train, pre_train, lvl_train])
val = pd.concat([ltt_val, cvi_val, pre_val, lvl_val])
test = pd.concat([ltt_test, cvi_test, pre_test, lvl_test])

# Shuffle the merged sub-dataframe 
train = train.sample(frac=1)
val = val.sample(frac=1)
test = test.sample(frac=1)

# Split the train, val, and test dataframes into "triples" and "weights"
train_triples = train[['head', 'variable', 'value']].to_numpy()
train_weights = train[['weights']].to_numpy()

val_triples = val[['head', 'variable', 'value']].to_numpy()
val_weights = val[['weights']].to_numpy()

test_triples = test[['head', 'variable', 'value']].to_numpy()
test_weights = test[['weights']].to_numpy()


# Create an dataframe "data" from the "train_triples". From these data frame we will create a list
# of unique entites (head and tail entities) and relations (l_text_topics, concept_vocab_index, prerequisite, level).
data = pd.DataFrame(train_triples, columns=['head', 'variable', 'value'])

data['head'].unique()
data['value'].unique()
data['variable'].unique()

data['head'].nunique()
data['value'].nunique()
data['variable'].nunique()

# "entities" and "relations" contains the list of unique entites and relations present in the train data.
# We will store the these entites, relations, and their corresponding embeddings. And will use these 
# further while calculating cosine similarities.
entities = data['head'].unique()
entities = np.append(entities, data['value'].unique())
# print(entities.shape)
entities = np.unique(entities)
# print(entities.shape)
relations = data['variable'].unique()
relations.shape

# Create and empty dataframe "hyperparameter_result". We will store the results of each model in this dataframe.
# It contains the columns: ['model_name', 'epochs', 'batches_count', 'k', 'structural_wt', 'lr','start_loss','end_loss', 'mrr' ,'mr', 'hits_10', 'hits_5', 'hits_3', 'hits_1', 'losses_list']
hyperparameter_result=pd.DataFrame(columns=['model_name', 'epochs', 'batches_count', 'k', 'structural_wt', 'lr','start_loss','end_loss', 'mrr' ,'mr', 'hits_10', 'hits_5', 'hits_3', 'hits_1', 'losses_list'])

"""
Below function takes as input the ModelName and its corresponding hyperparameter on which it needs to be trained.
It also takes as default parameter the unique entities and relations created earlier.

Based on the modelname (i.e. transE or holE), the function will store the outputs in the respective 
directory of that model.

Inside the model directory, it will create a sub-directories with the name similar to "transE_50_5_50_0.1_0.1" or "holE_50_5_50_0.1_0.1" based on 
the modelname and its corresponding parameters.
"""
def train_model(modelname, epochs, batches_count, k, structural_wt, lr, entities=entities, relations=relations):

    # model.fit will return the list of losses for each epoch
    if(modelname=="transE"):        
        model = TransE(batches_count=batches_count, seed=555, epochs=epochs, k=k, loss='pairwise', loss_params={'margin':5}, verbose = True, embedding_model_params={'structural_wt':structural_wt}, optimizer_params = {'lr':lr})
        losses_list= model.fit(train_triples, focusE_numeric_edge_values=train_weights)
        path = 'embeddings_final/transE/'
    elif(modelname=="holE"):
        model = HolE(batches_count=batches_count, seed=555, epochs=epochs, k=k, eta=5, loss='pairwise', loss_params={'margin':1}, regularizer='LP', regularizer_params={'lambda':0.1}, verbose=True, embedding_model_params={'structural_wt':structural_wt}, optimizer_params = {'lr':lr})
        losses_list= model.fit(train_triples, focusE_numeric_edge_values=train_weights)
        path = 'embeddings_final/holE/'


    dirname = modelname+"_"+str(epochs)+"_"+str(batches_count)+"_"+str(k)+"_"+str(structural_wt)+"_"+str(lr)
    path = path+dirname
    os.mkdir(path)    
    
    # Store the train triples in case we need it at later stage
    data.to_csv(path+'/train_triples.csv')

    # Get the embeddings for entities and relations from the model trained
    entity_embeddings = model.get_embeddings(entities, embedding_type='entity')
    relation_embeddings = model.get_embeddings(relations, embedding_type='relation')
    
    # Store the unique entities, relations and its corresponding embeddings into dataframe and save it in csv format
    entities = pd.DataFrame(entities)
    entity_embeddings = pd.DataFrame(entity_embeddings)
    entities.to_csv(path+'/entities.csv', index=False, header=False)
    entity_embeddings.to_csv(path+'/entity_embeddings.csv', index=False, header=False)
    
    relations = pd.DataFrame(relations)
    relation_embeddings = pd.DataFrame(relation_embeddings)
    relations.to_csv(path+'/relations.csv', index=False, header=False)
    relation_embeddings.to_csv(path+'/relation_embeddings.csv', index=False, header=False)
    
    
    
    filter = np.concatenate((train_triples, val_triples, test_triples))
    
    # Evaluate the model on the val data and get the different score values such as
    # mrr, mr, hits@10, hits@5, hits@3, hits@1
    ranks = evaluate_performance(val_triples, model = model, filter_triples = filter, use_default_protocol=True, verbose=True)
    mrr = mrr_score(ranks)
    mr = mr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    # print("MRR: %f, MR: %f, Hits@10: %f" % (mrr, mr, hits_10))
    hits_5 = hits_at_n_score(ranks, n=5)
    # print("Hits@5: %.2f" % (hits_5))
    hits_3 = hits_at_n_score(ranks, n=3)
    # print("Hits@3: %.2f" % (hits_3))
    hits_1 = hits_at_n_score(ranks, n=1)
    # print("Hits@1: %.2f" % (hits_1))

    # Append the evaluate results for each trained model into the "hyperparameter_result" dataframe created earlier.
    start_loss = losses_list[0]
    end_loss = losses_list[-1]
    row= [modelname, epochs, batches_count, k, structural_wt, lr,start_loss, end_loss, mrr ,mr,hits_10,hits_5,hits_3,hits_1, losses_list]
    hyperparameter_result.loc[len(hyperparameter_result)]=row

# To train a single model with only 1 set of hyperparamter uncomment the below line
# train_model("transE", 5000, 5, 50, 0.8, 0.001)
#train_model("holE", 5000, 5, 50, 0.8, 0.001)


# To train multiple models with multiple hyperparamter combinations, include the desired value
# in the below lists of hyperparameters
lrs=[0.1, 0.01, 0.05]           #Learning Rate
embed_size=[40, 50]             #Embedding size k
epochs=[100, 500, 1000]         #Epochs
model_name=['transE', 'holE']   #ModelNames


# Iterate over each of the above lists and train the model for each possible 
# permutations of the hyperparameters. Results of each such model will be stored in the
# "transE_holE_hyperparam_result.csv"
for model in model_name:
    for lr in lrs:
        for es in embed_size:
            for epoch in epochs:
                train_model(model,epoch, 5, es, 0.1, lr)

hyperparameter_result.to_csv('embeddings_final/output/transE_holE_hyperparam_result.csv')






