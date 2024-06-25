-- data_intermediates: Contains intermediate preprocessed csv files

    adf_pc_1.csv - This file containes the preprocessed data which we have used in our tasks

    v2_df.csv - Concept vocab list

    adf_with_tfidf.csv - This file is generated from the "tfidf_weights_for_concept_vocab.ipynb". It contains the tfidf score as weight for concept vocab words

    tw_df1.csv - This file is generated from the "triples_creation_and_merging.ipynb". It containes the triples along with the corresponding weight value.


-- GPT: Contains files that generate concept vocab from GPT using OpenAI api

    gpt_tst.ipynb: Has the prompt and queries.

    gpt.ipynb: processes the response txt.

    gpt3.5-ans.txt: the processed answers from GPT are stored here.

    gpt3.5-response.txt: responses from GPT.

tfidf_weights_for_concept_vocab.ipynb - This file is used to generate the "adf_with_tfidf.csv"

triples_creation_and_merging.ipynb - This file is used to generate the "triples_with_prob_tfidf.csv"

triples.csv: Generated triples with weights(prob) are stored here.

vocab_master_list.csv: contains, master list of all the concept vocab words.

