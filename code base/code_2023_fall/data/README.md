# Data Directory

This directory layout provides crucial resources for processing and analyzing data at different stages of the project.

# Preprocessing Raw Data

To preprocess raw data using the `preprocess.py` script, follow these steps:

1. Open a command prompt.

2. Run the following command to initiate the preprocessing:

    ```bash
    python preprocess.py
    ```

If you want to generate intermediate files for debugging purposes, you can enable debug mode by adding the `--debug` option:

```bash
python preprocess.py --debug
```

## `data_1_folder`

The `data_1_folder` directory contains the raw data, i.e. transcripts and file names corresponding to course names. This directory, along with the `vocab_master_list.csv` file, is essential for running the `preprocess.py` script.


## `debug_output`

Contains intermediate preprocessed CSV files:

- **adf_pc_1.csv**
  - Contains preprocessed data used in our tasks.

- **v2_df.csv**
  - Concept vocabulary list.

- **adf_with_tfidf.csv**
  - Generated from `tfidf_weights_for_concept_vocab.ipynb`. Contains TF-IDF scores as weights for concept vocabulary words.

- **tw_df1.csv**
  - Generated from `triples_creation_and_merging.ipynb`. Contains triples along with corresponding weight values.

## `GPT`

Contains files related to generating concept vocabulary from GPT using OpenAI API:

- **gpt_tst.ipynb**
  - Contains the prompt and queries for GPT.

- **gpt.ipynb**
  - Processes the response text from GPT.

- **gpt3.5-ans.txt**
  - Stores the processed answers from GPT.

- **gpt3.5-response.txt**
  - Contains responses from GPT.

## `Experiments`

Contains files that were used for experimentation, these can be ignored and perform no particular function other than for reference purposes:

- **tfidf_weights_for_concept_vocab.ipynb**
  - Used to generate `adf_with_tfidf.csv`.

- **triples_creation_and_merging.ipynb**
  - Used to generate `triples_with_prob_tfidf.csv`.


## Files:

- **preprocess.py**
  - Preprocess script, converts raw data to triples directly.

- **triples.csv**
  - Contains generated triples with weights (probability).

- **vocab_master_list.csv**
  - Contains the master list of all concept vocabulary words. 

