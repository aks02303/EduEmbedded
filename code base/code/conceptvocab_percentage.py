
"""
input: embeddings_final/input/entity1-entity2-cv.csv

output: 
    embeddigns_final/output/entities-cv-percentage.csv
    embeddigns_final/input/CV-entity1.csv
    embeddigns_final/input/CV-entity2.csv

Note: please change the paths to support your directory structure

"""

import pandas as pd
import json
import ast

# Load the "entity1-entity2-cv.csv" file. This file contains following columns:
# entity1, ltt1, cv1, entity2, ltt2, cv2
# In this file the entity1 and entity2 are such that ltt1 and ltt2 are similar (ltt -> l_text_topics)
# This csv file was manually created
df = pd.read_csv("embeddings_final/input/entity1-entity2-cv.csv")

# Load the "entities.csv" file from the desired model directory (i.e. transH/transH_50_5_50_0.1_0.1 in this case)
entities = list(pd.read_csv('embeddings_final/transH/transH_50_5_50_0.1_0.1/entities.csv', header=None)[0])


# The manually created file might have certain entities which are not present in the training triples.
# So, we need to remove such triples.
rm_entities = []
for i in range(len(df)):
    if(df.loc[i]["entity1"] not in entities):
        rm_entities.append(df.loc[i]["entity1"])

for i in rm_entities:
    df = df[df.entity1 != i]
df = df.reset_index(drop=True)

# List of concept_vocab_index in the cv1 and cv2 columns is in string format, so we need to convert it to list
df["cv1"] = df["cv1"].apply(lambda x:ast.literal_eval(x))
df["cv2"] = df["cv2"].apply(lambda x:ast.literal_eval(x))
df["ltt1"] = df["ltt1"].apply(lambda x:ast.literal_eval(x))
df["ltt2"] = df["ltt2"].apply(lambda x:ast.literal_eval(x))

cv1 = df["cv1"]
cv2 = df["cv2"]


# Iterate over all the rows of the find the percentage of common concept_vocab_index (cv1 and cv2) 
# for both the entities (courses)

temp = []
common_cv = []
for i in range(len(df)):

    total_len = len(set(df.loc[i]["cv1"]).union(set(df.loc[i]["cv2"])))
    common = list(set(df.loc[i]["cv1"]).intersection(set(df.loc[i]["cv2"])))

    common_new = []
    for c in common:
        #print(c)
        if c in entities:
            common_new.append(c)

    common_cv.append(common_new)
    common_len = len(set(df.loc[i]["cv1"]).intersection(set(df.loc[i]["cv2"])))
    per_sim = ( common_len / total_len ) * 100
    temp.append(per_sim)


df["common"] = common_cv
df["common_perc"] = temp


# create "df_left" and "df_right" dataframes which will contain
# (entity1 and common concept_vocab_index) and (entity2 and common concept_vocab_index) respectively
df_left = df[["entity1", "common"]]
df_left = df_left.explode("common")
df_left["variable"] = "concept_vocab_index"
df_left.rename(columns = {'entity1':'head', 'common':'value'}, inplace = True)
df_left = df_left[["head","variable","value"]]

df_right = df[["entity2", "common"]]
df_right = df_right.explode("common")
df_right["variable"] = "concept_vocab_index"
df_right.rename(columns = {'entity2':'head', 'common':'value'}, inplace = True)
df_right = df_right[["head","variable","value"]]

# For final output we get 3 files:
# entities-cv-percentage.csv : this is same as "entity1-entity2-cv.csv" with 1 column (common_perc) added 
# cv-entity1.csv : contains the entity1 and its corresponding ltt1 and common concept_vocab_index (cv present in both entity1 and entity2)
# cv-entity2.csv : contains the entity2 and its corresponding ltt2 and common concept_vocab_index (cv present in both entity1 and entity2)

df.to_csv("embeddings_final/output/entities-cv-percentage.csv")
df_left.to_csv("embeddings_final/input/CV-entity1.csv")
df_right.to_csv("embeddings_final/input/CV-entity2.csv")
