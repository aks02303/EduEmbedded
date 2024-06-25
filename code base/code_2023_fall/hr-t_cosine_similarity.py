#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file calculates cosine similarity between h+r and t for any triples

input:
    embeddings_final/input/LTT_entity1.csv OR
    embeddings_final/input/LTT_entity2.csv OR
    embeddings_final/input/CV-entity1.csv  OR
    embeddings_final/input/CV_entity2.csv  OR Any other file having triples in the format head, variable, value


output:
    embeddigns_final/{model/modelName}/entity1-LTT-head-rel-tail-cos-sim.csv
                    "                    /entity2-LTT-head-rel-tail-cos-sim.csv
                    "                    /entity1-CV-head-rel-tail-cos-sim.csv
                    "                    /entity2-CV-head-rel-tail-cos-sim.csv

Note: please change the path and name of the input and output files as per your requirement

"""


from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

# give the path of the model directory for which we want to find the cosine similarity
path = 'embeddings_final/transH/transH_50_5_50_0.1_0.1/'
#path = 'embeddings_final/transE/transE_50_5_50_0.1_0.1/'


# Give the path of the input csv which contains the triples in the form of (head, variable, value)
# This file could be LTT_entity1/2, CV-entity1/2, or any other file in similar format.
triples = pd.read_csv('embeddings_final/input/CV-entity2.csv')
#triples = pd.read_csv('embeddings_final/input/LTT_entity2.csv')


# Load the entities, relations, and their corresponding embeddings
entities = pd.read_csv(path + 'entities.csv', header=None)
entities_embeddings = pd.read_csv(path + 'entity_embeddings.csv', header=None)

relations = pd.read_csv(path + 'relations.csv', header=None)
relations_embeddings = pd.read_csv(path + 'relation_embeddings.csv', header=None)



X = triples

# This function takes 2 array of embeddings (h+r and t) as input and returns its cosine similarity
def cos_sim_hrt(hr, tail):
    print("similarity score : ", cosine_similarity(hr, tail))
    return cosine_similarity(hr, tail)[0][0]
    

# Create cos_sim_df dataframe which will contain the input triples along with the 
# cosine similarity score between h+r and t
cos_sim_df = pd.DataFrame(columns = ['head','relation','tail','cos_sim_score'])

# Iterate over all the rows of the input triples 
for i in range(len(X)):

    # Find the index of the entity in the "entities" and index of the relation in the "relations"
    headindex = entities.index[entities[0]==X.iloc[i]['head']].tolist()
    tailindex = entities.index[entities[0]==X.iloc[i]['value']].tolist()
    relindex = relations.index[relations[0]==X.iloc[i]['variable']].tolist()

    print(X.iloc[i]['head'])
    print(entities[0][headindex])

    print(X.iloc[i]['value'])
    print(relations[0][relindex])
    

    print(X.iloc[i]['variable'])
    print(entities[0][tailindex])

    # Get the entity value from the above fetched index
    head = entities[0][headindex].values[0]
    rel = relations[0][relindex].values[0]
    tail = entities[0][tailindex].values[0]
    
    # Get the embeddings corresponding to the above fetched index from
    # "entities_embeddings" and "relations_embeddings"
    headembedding = np.array(entities_embeddings.iloc[headindex])
    relembedding = np.array(relations_embeddings.iloc[relindex])
    tailembedding = np.array(entities_embeddings.iloc[tailindex])

    # Head + relation
    hrembedding = headembedding + relembedding

    # Call the function to get the cosine similarity between head+rel and tail
    cos_sim_score = cos_sim_hrt(hrembedding, tailembedding)
    row = [head, rel, tail, cos_sim_score]

    # Append the triples with the cos score value in the "cos_sim_df"
    cos_sim_df.loc[len(cos_sim_df)] = row


# Save the dataframe in csv
cos_sim_df.to_csv(path + 'entity2-CV-head-rel-tail-cos-sim.csv')
#cos_sim_df.to_csv(path + 'entity2-LTT-head-rel-tail-cos-sim.csv')

# entities[0][53]
# entities[0][302]
# relations[0][3]
# head = np.array([entities_embeddings.iloc[53]])
# rel = np.array([relations_embeddings.iloc[3]])
# tail = np.array([entities_embeddings.iloc[302]])
# hr = head+rel
# cos_sim_hrt(hr, tail)

# entities[0][184]
# entities[0][304]
# relations[0][3]
# head = np.array([entities_embeddings.iloc[184]])
# rel = np.array([relations_embeddings.iloc[3]])
# tail = np.array([entities_embeddings.iloc[304]])
# hr = head+rel
# cos_sim_hrt(hr, tail)


# entities[0][285]
# entities[0][486]
# relations[0][0]
# head = np.array([entities_embeddings.iloc[285]])
# rel = np.array([relations_embeddings.iloc[0]])
# tail = np.array([entities_embeddings.iloc[486]])
# hr = head+rel
# cos_sim_hrt(hr, tail)
# """
# head = Course3_W7-S1-L2_-_Part_of_Speech_Tagging_-_15_slides_18-08	
# relation = concept_vocab_index	
# tail = vi1422
# similarity score: [[0.99993563]]
# """

# entities[0][55]
# entities[0][307]
# relations[0][1]
# head = np.array([entities_embeddings.iloc[55]])
# rel = np.array([relations_embeddings.iloc[1]])
# tail = np.array([entities_embeddings.iloc[307]])
# hr = head+rel
# cos_sim_hrt(hr, tail)
# """
# head = Course1_W4-S2-L5_Parameter_Estimation_in_Lexicalized_PCFGs_Part_2_9-08	
# relation = l_text_topics	
# tail = topic_1
# similarity score: [[0.9999752]]
# """
