# Code 2023 Fall directory

Please refer to the code hierarchy below to gain an understanding of the project workflow:

## `code`

### `data`


- **triples_gen.ipynb**
    - Contains code to preprocess data, perform LDA, NMF, triples generation, etc. Output files: `adf_pc_1.csv`, `tw_df1.csv`.

- **tfidf_weights_for_concept_vocab.ipynb**
    - Generates `adf_with_tfidf.csv`.

- **triples_creation_and_merging.ipynb**
    - Generates `triples_with_prob_tfidf.csv`.

- **triples.csv**
    - Final data used for training the KGE models.

- `data/data_intermediates`
    <details>
    <summary>Files</summary>

    - **adf_pc_1.csv**
        - Contains preprocessed data used as our base data.

    - **v2_df.csv**
        - Concept vocabulary list.

    - **tw_df1.csv**
        - Used to generate triples for topics-concept_vocab relation.

    - **adf_with_tfidf.csv**
        - Contains TF-IDF scores as weights for concept vocab words.
    </details>

### `eduTransE_HolE`

- Directory containing package files for TransE and HolE. This directory contains package files related to TransE and HolE. Package by default does not provide losses for each epoch. The package is modified to return the losses for each epoch which we are further using for analysis and graph creation. 

- **NOTE**: DON'T IMPORT AMPLIGRAPH PACKAGE, INSTEAD USE MODULES FROM THIS DIRECTORY.

### `eduTransH`

- Directory containing package files for TransH. To train transH model run the following command from **eduTransH** directory:
    ```python
    python -m latent_features.driver4
    ```        

### `embeddings_final`

- Directory containing KGE model specific files.

    - `transH`, `holE`, `transE`
        - Directories for specific models with hyperparameters.

-  Directories based on model hyperparameter will be automatically created and respective output files will be stored in respective model's directory

- `embeddings_final/transH`

    - `transH_50_5_50_0.1_0.1` (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
        <details>
        <summary>Files</summary>

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
        </details>

- `embeddings_final/holE`
    - `holE_50_5_50_0.1_0.1` (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
        <details>
        <summary>Files</summary>

        - entities.csv
        - entity_embeddings.csv
        - relations.csv
        - relation_embeddings.csv
        - train_triples.csv
        - holEsimilar_entities_cos_sim.csv
        - holEdiff_entities_cos_sim.csv
        </details>

- `embeddings_final/transE`
    - `transE_50_5_50_0.1_0.1`
        <details>
        <summary>Files</summary>

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
     </details>

- `embeddings_final/input` (Files in this directory are used by the cosine_similiarty related .py files as input for generating output files with 
                variations of entities and relations)
    <details>
    <summary>Files</summary>

    - similar_entities_cos_sim.csv (Input for calculating cosine similarity between 2 similar entities. This file was manually created.)

    - diff_entities_cos_sim.csv (Input for calculating cosine similarity between 2 different entities. This file was manually created.)

    - LTT_entity1.csv (This file contains triples of relation l_text_topics. Created manually.)

    - LTT_entity2.csv (This file contains triples of relation l_text_topics. Created manually.)
        LTT_entity1.csv and LTT_entity2.csv both contains different courses (head) having same corresponding topics value (tail)

    - entity1-entity2-cv.csv (This file contains course, l_text_topics, concept_vocab_word for 2 different list of courses (entity1 and entity2))

    - CV-entity1.csv (This file contains triples of relation concept_vocab_index. Generated from concept_vocab_percentage.py)

    - CV-entity2.csv (This file contains triples of relation concept_vocab_index. Generated from concept_vocab_percentage.py)
        CV-entity1.csv and CV-entity2.csv both contains different courses (head) having same corresponding concept_vocab_index value (tail)
    </details>

- `embeddings_final/output` 
    <details>
    <summary>Files</summary>

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
    </details>

### `Files`

- *embedding_generation.py*

    - Generates embeddings and hyperparameter results for TransE and HolE.

    - input: data/triples.csv


    - output: 

        embeddings_final/transE/transE_50_5_50_0.1_0.1 (modelName_epochs_batchescount_embeddingsize_structuralweight_learningrate)
            - entities.csv
            - entity_embeddings.csv
            - relations.csv
            - relation_embeddings.csv
            - train_triples.csv

        output/transE_holE_hyperparam_result.csv

    Refer file to know how to train multiple models with various hyperparameter.


- *graphs.py*

    - Generates loss vs epoch graphs.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input: embeddings_final/output/transE_holE_hyperparam_result.csv OR embeddings_final/output/transH_hyperparam_result.csv
        
        - output: embeddings_final/output/transE_holE_hyperparam_result.pdf OR 
                    embeddings_final/output/transH_hyperparam_result.pdf
    </details>


- **concept_vocab_percentage.py**

    - Calculates the percentage similarity of concept_vocab_index between different entities.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input: embeddings_final/input/entity1-entity2-cv.csv

        - output: 
            embeddigns_final/output/entities-cv-percentage.csv
            embeddigns_final/input/CV-entity1.csv
            embeddigns_final/input/CV-entity2.csv
    
    </details>

- **cosine_similarity.py**

    - Tests the cosine similarity of specific embeddings.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input: 
            embeddings_final/model/modelName/entites.csv
            embeddings_final/model/modelName/entity_embeddings.csv
            embeddings_final/model/modelName/relation.csv
            embeddings_final/model/modelName/relation_embeddings.csv

        - output:
            similarity score
    
    </details>

- **hr-t_cosine_similarity.py**

    - Calculates cosine similarity between h+r and t for any triples.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input:
            embeddings_final/input/LTT_entity1.csv OR
            embeddings_final/input/LTT_entity2.csv OR
            embeddings_final/input/CV-entity1.csv  OR
            embeddings_final/input/CV_entity2.csv  OR Any other file having triples in the format head, variable, value


        - output:
            embeddigns_final/{model/modelName}/entity1-LTT-head-rel-tail-cos-sim.csv
                            "                    /entity2-LTT-head-rel-tail-cos-sim.csv
                            "                    /entity1-CV-head-rel-tail-cos-sim.csv
                            "                    /entity2-CV-head-rel-tail-cos-sim.csv

    
    </details>

- **hr-t_e1-e2_cosine_similarity.py**

    - Calculates cosine similarity between h+r and t for any triples and cosine similarity between any two entities.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input1:
                embeddings_final/input/LTT_entity1.csv OR
                embeddings_final/input/LTT_entity2.csv OR
                embeddings_final/input/CV-entity1.csv  OR
                embeddings_final/input/CV_entity2.csv  OR Any other file having triples in the format head, variable, value
        - input2:
                embeddings_final/input/similar_entities_cos_sim.csv  OR
                embeddings_final/input/diff_entities_cos_sim.csv  OR Any other file having 2 entities in the format entity1, entity2

        - output1:
                embeddigns_final/{model/modelName}/entity1-LTT-head-rel-tail-cos-sim.csv
                             "                    /entity2-LTT-head-rel-tail-cos-sim.csv
                             "                    /entity1-CV-head-rel-tail-cos-sim.csv
                             "                    /entity2-CV-head-rel-tail-cos-sim.csv
        - output2:
                embeddigns_final/{model/modelName}/similar_entities_cos_sim.csv
                             "                    /diff_entities_cos_sim.csv

    
    </details>

- **hr-hr_cosine_similarity.py**

    - Calculates cosine similarities between h+r of entity1 and h+r of entity2.

    - <details>
        <summary> Input/Ouput </summary>
        
        - input1:
            embeddings_final/input/LTT_entity1.csv and
            embeddings_final/input/LTT_entity2.csv 
                        OR
            embeddings_final/input/CV-entity1.csv  and
            embeddings_final/input/CV_entity2.csv  
        - output:
            embeddigns_final/{model/modelName}/LTT-entity1-entity2-head-rel-cos-sim.csv
                        OR
            embeddigns_final/{model/modelName}/CV-entity1-entity2-head-rel-cos-sim.csv

    
    </details>

- **manual_generated_data.xlsx**

    - Contains various sheets used to create manually generated input files.

**Note:** Please adjust the file paths in the code to match your directory hierarchy.

```plaintext

root
    folder1
        subfolder1
        file1.txt
        subfolder2
        file2.txt
    folder2
        file3.txt
```