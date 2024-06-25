
Please refer below code hierarchy to get the understanding of the project workflow.


- code 
    - data
        - triples_gen.ipynb (code file provided to us which contained all the steps to preprocess the data, lda, nmf, triples,etc. 
            Output file is adf_pc_1.csv, tw_df1.csv) 

        - tfidf_weights_for_concept_vocab.ipynb (This file is used to generate the "adf_with_tfidf.csv")

        - triples_creation_and_merging.ipynb - This file is used to generate the "triples_with_prob_tfidf.csv"

        - adf_pc_1.csv - This file containes the preprocessed data which we have used as our base data to start from

        - v2_df.csv - Concept vocab list

        - tw_df1.csv - This file is used to generate triples for topics concept_vocab relation
        
        - adf_with_tfidf.csv - This file is generated from the "tfidf_weights_for_concept_vocab.ipynb". It contains the tfidf score as weight for concept vocab words

        - triples.csv - This file is generated from the "triples_creation_and_merging.ipynb". It containes the triples along with the corresponding weight value. This is the final data that we will use for training the KGE models.


    - eduTransE_HolE
        This directory contains package files related to TransE and HolE. Package by default is not providing losses for each epoch. We modified the package to
        return the losses for each epoch which we are further using for analysis and graph creation. DON'T IMPORT AMPLIGRAPH PACKAGE, INSTEAD USE MODULES FROM THIS DIRECTORY.

    - eduTransH 
        This directory contains package files related to TransH.
        To train transH model run the following command from this directory:
        python -m latent_features.driver4

    - embeddings_final (This directory will contain the KGE model specific files. Directories based on model hyperparameter will be automatically created and 
                        respective output files will be stored in respective model's directory)

        - transH 
            -transH_50_5_50_0.1_0.1 (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
                - entities.csv
                - entity_embeddings.csv
                - relations.csv
                - relation_embeddings.csv
                - train_triples.csv
                - transH_hrt_cos_sim.csv
                - transHsimilar_entities_cos_sim.csv
                - transHdiff_entities_cos_sim.csv
                - entity1-LTT-head-rel-tail-cos-sim.csv
                - entity2-LTT-head-rel-tail-cos-sim.csv
                - LTT-entity1-entity2-head-rel-cos-sim.csv
                - entity1-CV-head-rel-tail-cos-sim.csv
                - entity2-CV-head-rel-tail-cos-sim.csv
                - CV-entity1-entity2-head-rel-cos-sim.csv
            .
            .

        - holE
            -holE_50_5_50_0.1_0.1 (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
                - entities.csv
                - entity_embeddings.csv
                - relations.csv
                - relation_embeddings.csv
                - train_triples.csv
                - holEsimilar_entities_cos_sim.csv
                - holEdiff_entities_cos_sim.csv

            .
            .

        - transE 
            -transE_50_5_50_0.1_0.1
                - entities.csv
                - entity_embeddings.csv
                - relations.csv
                - relation_embeddings.csv
                - train_triples.csv
                - transE_hrt_cos_sim.csv
                - transEsimilar_entities_cos_sim.csv
                - transEdiff_entities_cos_sim.csv
                - entity1-LTT-head-rel-tail-cos-sim.csv
                - entity2-LTT-head-rel-tail-cos-sim.csv
                - LTT-entity1-entity2-head-rel-cos-sim.csv
                - entity1-CV-head-rel-tail-cos-sim.csv
                - entity2-CV-head-rel-tail-cos-sim.csv
                - CV-entity1-entity2-head-rel-cos-sim.csv
            .
            .


        - input (Files in this directory are used by the cosine_similiarty related .py files as input for generating output files with 
                variations of entities and relations)

            - similar_entities_cos_sim.csv (Input for calculating cosine similarity between 2 similar entities. This file was manually created.)

            - diff_entities_cos_sim.csv (Input for calculating cosine similarity between 2 different entities. This file was manually created.)

            - LTT_entity1.csv (This file contains triples of relation l_text_topics. Created manually.)

            - LTT_entity2.csv (This file contains triples of relation l_text_topics. Created manually.)
                LTT_entity1.csv and LTT_entity2.csv both contains different courses (head) having same corresponding topics value (tail)

            - entity1-entity2-cv.csv (This file contains course, l_text_topics, concept_vocab_word for 2 different list of courses (entity1 and entity2))

            - CV-entity1.csv (This file contains triples of relation concept_vocab_index. Generated from concept_vocab_percentage.py)

            - CV-entity2.csv (This file contains triples of relation concept_vocab_index. Generated from concept_vocab_percentage.py)
                CV-entity1.csv and CV-entity2.csv both contains different courses (head) having same corresponding concept_vocab_index value (tail)


        - output 

            - entites-cv-percentage.csv (contains the percentage value to describe how much similar is the concept_vocab_index of 2 different entities)
                    (input : entity1-entity2-cv.csv)

            - transE_holE_hyperparam_result.csv (transE and holE output results for various hyperparameters)
                    (model_name	epochs	batches_count	k	structural_wt	lr	start_loss	end_loss	mrr	mr	hits_10	hits_5	hits_3	hits_1	losses_list)

            - transE_holE_hyperparam_result.pdf
                    (loss vs epoch graphs generated using "transE_holE_hyperparam_result.csv")

            - transH_hyperparam_result.csv (transH output results for various hyperparameters)
                    (model_name	epochs	batches_count	k	structural_wt	lr	start_loss	end_loss	mrr	mr	hits_10	hits_5	hits_3	hits_1	losses_list)

            - transH_hyperparam_result.pdf
                    (loss vs epoch graphs generated using "transH_hyperparam_result.csv")

    - embedding_generation.py (Generates embeddings and hyperparameter results for transE and holE)
        {
            input: data/triples.csv


            output: 

                embeddings_final/transE/transE_50_5_50_0.1_0.1 (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
                    - entities.csv
                    - entity_embeddings.csv
                    - relations.csv
                    - relation_embeddings.csv
                    - train_triples.csv

                output/transE_holE_hyperparam_result.csv

            Refer file to know how to train multiple models with various hyperparameter.

        }

    - graphs.py (Generating loss vs epoch graphs)
        {
            input: embeddings_final/output/transE_holE_hyperparam_result.csv OR 
                    embeddings_final/output/transH_hyperparam_result.csv
            output: embeddings_final/output/transE_holE_hyperparam_result.pdf OR 
                    embeddings_final/output/transH_hyperparam_result.pdf
        }

    - concept_vocab_percentage.py
        {
            input: embeddings_final/input/entity1-entity2-cv.csv

            output: 
                embeddigns_final/output/entities-cv-percentage.csv
                embeddigns_final/input/CV-entity1.csv
                embeddigns_final/input/CV-entity2.csv
                
        }
    - cosine_similiarity.py (This file is for testing the specific embedding's cosine similarity)
        {
            input: 
                embeddings_final/model/modelName/entites.csv
                embeddings_final/model/modelName/entity_embeddings.csv
                embeddings_final/model/modelName/relation.csv
                embeddings_final/model/modelName/relation_embeddings.csv

            output:
                similarity score

        }

    - hr-t_cosine_similarity.py (This file calculates cosine similarity between h+r and t for any triples)
        {
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

        }

    - hr-t_e1-e2_cosine_similarity.py(This file calculates cosine similarity between h+r and t for any triples and cosine similarity between any 2 entities)
        {
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

        }

    - hr-hr_cosine_similariy.py (This file calculates cosine similarities between h+r of entity1 and h+r of entity2)
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

    - manual_generated_data.xlsx (This file contains various sheets used to create the manual generated input files)

Note: please change the paths in the file to suit your directory hierarchy.





