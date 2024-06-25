#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file calculates cosine similarity between h+r and t for any triples (transE and transhH) and 
cosine similarity between any 2 entities (transE, transH, holE)

input1:
    embeddings_final/input/LTT_entity1.csv OR
    embeddings_final/input/LTT_entity2.csv OR
    embeddings_final/input/CV-entity1.csv  OR
    embeddings_final/input/CV_entity2.csv  OR Any other file having triples in the format head, variable, value
input2:
    embeddings_final/input/similar_entities_cos_sim.csv  OR
    embeddings_final/input/diff_entities_cos_sim.csv  OR Any other file having 2 entities in the format entity1, entity2

output1:
    embeddigns_final/{model/modelName}/entity1-LTT-head-rel-tail-cos-sim.csv
                    "                    /entity2-LTT-head-rel-tail-cos-sim.csv
                    "                    /entity1-CV-head-rel-tail-cos-sim.csv
                    "                    /entity2-CV-head-rel-tail-cos-sim.csv
output2:
    embeddigns_final/{model/modelName}/similar_entities_cos_sim.csv
                    "                    /diff_entities_cos_sim.csv

Note: please change the path and file name as per your requirement

"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# path = 'embeddings_final/transE/'
# path = 'embeddings_final/holE'

# Change the model name and parameters for which we want to find the cos sim
model_type='holE'
parameter='holE_50_5_50_0.1_0.1'

path = 'embeddings_final/'+model_type+'/'+parameter+'/'
# path = 'embeddings_final/transE/transE_300_5_50_0.2_0.001/'

# Input csv of the form (head variable value) for calculating similarity between h+r and t
hrt_cos_sim_input = pd.read_csv('embeddings_final/input/LTT_entity1.csv')
# triples = pd.read_csv('embeddings_final/transE/input.csv')

# Input csv of the form (entity1 entity2) for calculating similarity between e1 and e2
entities_cos_sim_input = pd.read_csv('embeddings_final/input/diff_entities_cos_sim.csv')
#entities_cos_sim_input = pd.read_csv('embeddings_final/input/similar_entities_cos_sim.csv')

# Load the entities, relations, and their corresponding embeddings
entities = pd.read_csv(path + 'entities.csv', header=None)
entities_embeddings = pd.read_csv(path + 'entity_embeddings.csv', header=None)

relations = pd.read_csv(path + 'relations.csv', header=None)
relations_embeddings = pd.read_csv(path + 'relation_embeddings.csv', header=None)


# This functions takes 2 arrays as input and returns the similarity score between them
def cos_sim(input1, input2):
    print("similarity score : ", cosine_similarity(input1, input2))
    return cosine_similarity(input1, input2)[0][0]

# If model_type is NOT "holE" then execute below loop to calculate h+r and t similiarity
# and store output in the desired path
if(model_type!='holE'):
    X = hrt_cos_sim_input
        
    cos_sim_df = pd.DataFrame(columns = ['head','relation','tail','cos_sim_score'])
    for i in range(len(X)):
        headindex = entities.index[entities[0]==X.iloc[i]['head']].tolist()
        tailindex = entities.index[entities[0]==X.iloc[i]['value']].tolist()
        relindex = relations.index[relations[0]==X.iloc[i]['variable']].tolist()

        print(X.iloc[i]['head'])
        print(entities[0][headindex])

        print(X.iloc[i]['value'])
        print(relations[0][relindex])
        

        print(X.iloc[i]['variable'])
        print(entities[0][tailindex])

        head = entities[0][headindex].values[0]
        rel = relations[0][relindex].values[0]
        tail = entities[0][tailindex].values[0]
        
        headembedding = np.array(entities_embeddings.iloc[headindex])
        relembedding = np.array(relations_embeddings.iloc[relindex])
        tailembedding = np.array(entities_embeddings.iloc[tailindex])
        hrembedding = headembedding + relembedding
        cos_sim_score = cos_sim(hrembedding, tailembedding)
        row = [head, rel, tail, cos_sim_score]
        cos_sim_df.loc[len(cos_sim_df)] = row

    cos_sim_df.to_csv(path +model_type+ '_hrt_cos_sim_output.csv')

# Irrespective of the model_type execute below loop to calculate e1 and e2 similarity score
# and store output in the desired path
Y = entities_cos_sim_input
cos_sim_entities_df = pd.DataFrame(columns = ['entity1','entity2','cos_sim_score'])
for i in range(len(Y)):
    print(i)
    entity1index = entities.index[entities[0]==Y.iloc[i]['entity1']].tolist()
    entity2index = entities.index[entities[0]==Y.iloc[i]['entity2']].tolist()
    
    entity1 = entities[0][entity1index].values[0]
    entity2 = entities[0][entity2index].values[0]

    print(entity1index)
    print(entity1)
    print(entity2index)
    print(entity2)

    entity1embedding = np.array(entities_embeddings.iloc[entity1index])
    entity2embedding = np.array(entities_embeddings.iloc[entity2index])

    print(entity1embedding)
    print(entity2embedding)

    cos_sim_score = cos_sim(entity1embedding,entity2embedding)
    row = [entity1,entity2, cos_sim_score]
    cos_sim_entities_df.loc[len(cos_sim_entities_df)] = row

cos_sim_entities_df.to_csv(path +model_type+ 'diff_entities_cos_sim.csv')


