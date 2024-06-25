#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is for testing the specific embedding's cosine similarity
input: 
    embeddings_final/model/modelName/entites.csv
    embeddings_final/model/modelName/entity_embeddings.csv
    embeddings_final/model/modelName/relation.csv
    embeddings_final/model/modelName/relation_embeddings.csv

output:
    similarity score

"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

entities = pd.read_csv('embeddings_final/transE/transE_50_5_50_0.1_0.1entities.csv', header=None)
entities_embeddings = pd.read_csv('embeddings_final/transE/transE_50_5_50_0.1_0.1/entity_embeddings.csv', header=None)

relations = pd.read_csv('embeddings_final/transE/transE_50_5_50_0.1_0.1/relations.csv', header=None)
relations_embeddings = pd.read_csv('embeddings_final/transE/transE_50_5_50_0.1_0.1/relation_embeddings.csv', header=None)

relations[0][0]
relations_embeddings.iloc[0]

entities[0][6]
entities[0][7]
entities[0][40]
entities[0][33]


def cos_sim(e1, e2):

    a1 = np.array([entities_embeddings.iloc[e1]])
    a2 = np.array([entities_embeddings.iloc[e2]])
    
    b1 = np.array([relations_embeddings.iloc[0]])
    b2 = np.array([relations_embeddings.iloc[0]])
    
    sum1 = np.add(a1, b1)
    sum2 = np.add(a2, b2)
    print("head1 : ", entities[0][e1])
    print("head2 : ", entities[0][e2])
    print("relation : ", relations[0][0])
    print("similarity score : ", cosine_similarity(sum1, sum2))
    print()

def cos_sim_hrt(hr, tail):
    print("similarity score : ", cosine_similarity(hr, tail))

entities[0][285]
entities[0][486]
relations[0][0]
head = np.array([entities_embeddings.iloc[285]])
rel = np.array([relations_embeddings.iloc[0]])
tail = np.array([entities_embeddings.iloc[486]])
hr = head+rel
cos_sim_hrt(hr, tail)
"""
head = Course3_W7-S1-L2_-_Part_of_Speech_Tagging_-_15_slides_18-08	
relation = concept_vocab_index	
tail = vi1422
similarity score: [[0.99993563]]
"""

entities[0][55]
entities[0][307]
relations[0][1]
head = np.array([entities_embeddings.iloc[55]])
rel = np.array([relations_embeddings.iloc[1]])
tail = np.array([entities_embeddings.iloc[307]])
hr = head+rel
cos_sim_hrt(hr, tail)
"""
head = Course1_W4-S2-L5_Parameter_Estimation_in_Lexicalized_PCFGs_Part_2_9-08	
relation = l_text_topics	
tail = topic_1
similarity score: [[0.9999752]]
"""

a1 = np.array([entities_embeddings.iloc[0]])
a2 = np.array([entities_embeddings.iloc[1]])

b1 = np.array([relations_embeddings.iloc[0]])
b2 = np.array([relations_embeddings.iloc[0]])

sum1 = np.add(a1, b1)
sum2 = np.add(a2, b2)

# cos_sim(173,1)
# cos_sim(12,33)
# cos_sim(15,100)
# cos_sim(306, 329)
# cos_sim(310, 600)
# cos_sim(500, 800)
# cos_sim(88, 555)


