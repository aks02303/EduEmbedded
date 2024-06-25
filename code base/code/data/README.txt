adf_pc_1.csv - This file containes the preprocessed data which we have used in our tasks

v2_df.csv - Concept vocab list

adf_with_tfidf.csv - This file is generated from the "tfidf_weights_for_concept_vocab.ipynb". It contains the tfidf score as weight for concept vocab words

triples_with_prob_tfidf.csv - This file is generated from the "triples_creation_and_merging.ipynb". It containes the triples along with the corresponding weight value.

tfidf_weights_for_concept_vocab.ipynb - This file is used to generate the "adf_with_tfidf.csv"

triples_creation_and_merging.ipynb - This file is used to generate the "triples_with_prob_tfidf.csv"


embedding_generation.py - This file is used to generate the embeddings using transE and holE. It takes the "triples_with_prob_tfidf.csv" file as input

cosine_similarity.py - This file is used to compute the similarity score between embeddings

Note: please change the paths in the file to suit your directory hierarchy.
