#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file calculates cosine similarities between h+r of entity1 and h+r of entity2
input1:
    embeddings_final/input/LTT_entity1.csv and
    embeddings_final/input/LTT_entity2.csv 
                OR
    embeddings_final/input/CV-entity1.csv  and
    embeddings_final/input/CV_entity2.csv  
output:
    embeddigns_final/{model/modelName}/LTT-entity1-entity2-head-rel-cos-sim.csv
                OR
    embeddigns_final/{model/modelName}/CV-entity1-entity2-head-rel-cos-sim.csv

Note: please change the path and file name as per your requirement

"""

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

# Give path of the model for which we want to calculate the similarity score
path = 'embeddings_final/transH/transH_50_5_50_0.1_0.1/'
#path = 'embeddings_final/holE'

# Load 2 csv files which contains the triples. These head triples are such that they they have same l_text_topics
triplesleft = pd.read_csv('embeddings_final/input/CV-entity1.csv')
triplesright = pd.read_csv('embeddings_final/input/CV-entity2.csv')

#triplesleft = pd.read_csv('embeddings_final/input/LTT_entity1.csv')
#triplesright = pd.read_csv('embeddings_final/input/LTT_entity2.csv')

#Load the entities, relations, and their corresponding embeddings
entities = pd.read_csv(path + 'entities.csv', header=None)
entities_embeddings = pd.read_csv(path + 'entity_embeddings.csv', header=None)

relations = pd.read_csv(path + 'relations.csv', header=None)
relations_embeddings = pd.read_csv(path + 'relation_embeddings.csv', header=None)

XL = triplesleft
XR = triplesright

# Below functions takes 2 arrays as input and returns its cosine similarity
def cos_sim_hrt(hrleft, hrright):
    # print("similarity score : ", cosine_similarity(hrleft, hrright))
    return cosine_similarity(hrleft, hrright)[0][0]
    

# Iterate over both the triples simultaneously and find the head+relations embeddings for both of them
# Then find similarity between them.
cos_sim_df = pd.DataFrame(columns = ['head1','relation1','head2','relation2','cos_sim_score'])
for i in range(len(XL)):
    headindexleft = entities.index[entities[0]==XL.iloc[i]['head']].tolist()
    tailindexleft = entities.index[entities[0]==XL.iloc[i]['value']].tolist()
    relindexleft = relations.index[relations[0]==XL.iloc[i]['variable']].tolist()

    headindexright = entities.index[entities[0]==XR.iloc[i]['head']].tolist()
    tailindexright = entities.index[entities[0]==XR.iloc[i]['value']].tolist()
    relindexright = relations.index[relations[0]==XR.iloc[i]['variable']].tolist()

    headleft = entities[0][headindexleft].values[0]
    relleft = relations[0][relindexleft].values[0]
    tailleft = entities[0][tailindexleft].values[0]

    headright = entities[0][headindexright].values[0]
    relright = relations[0][relindexright].values[0]
    tailright = entities[0][tailindexright].values[0]


    headembeddingleft = np.array(entities_embeddings.iloc[headindexleft])
    relembeddingleft = np.array(relations_embeddings.iloc[relindexleft])
    tailembeddingleft = np.array(entities_embeddings.iloc[tailindexleft])
    hrembeddingleft = headembeddingleft + relembeddingleft

    headembeddingright = np.array(entities_embeddings.iloc[headindexright])
    relembeddingright = np.array(relations_embeddings.iloc[relindexright])
    tailembeddingright = np.array(entities_embeddings.iloc[tailindexright])
    hrembeddingright = headembeddingright + relembeddingright

    cos_sim_score = cos_sim_hrt(hrembeddingleft, hrembeddingright)
    row = [headleft, relleft, headright, relright, cos_sim_score]
    cos_sim_df.loc[len(cos_sim_df)] = row

# Save the final csv with similarity score 
cos_sim_df.to_csv(path + 'CV-entity1-entity2-head-rel-cos-sim.csv')
#cos_sim_df.to_csv(path + 'LTT-entity1-entity2-head-rel-cos-sim.csv')
