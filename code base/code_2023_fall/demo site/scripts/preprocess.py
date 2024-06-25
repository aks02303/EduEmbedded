# This file is a copy of the original preprocess.py file from the data directory, please refer to the original file for documentation

import pandas as pd 
import numpy as np 
from collections import defaultdict
from itertools import chain
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import os


def complete(func):
    def wrapper(*args):
        # print(f"{func.__name__}...", end='')
        ret_val = func(*args)
        # print('done')
        return ret_val
    return wrapper


@complete
def load_words(path_to_master_list):
    df = pd.read_csv(path_to_master_list, names=['0'], header=None)
    mstr_list = list(set(df['0'].tolist()))
    mstr_list_clean = [vocab for vocab in mstr_list if str(vocab) != 'nan']

    # replace space with dash
    separator = '_'
    word_list = [separator.join(word.split()) for word in mstr_list_clean]

    return mstr_list_clean, word_list

@complete
def load_test_csv(path_to_test_csv):
    df = pd.read_csv(path_to_test_csv)
    return df

@complete
def process_text(text):
    def _get_phrases(fle):
        phrase_dict = defaultdict(list)
        for line in map(str.rstrip, fle):
            k, _, phr = line.partition(" ")
            phrase_dict[k].append(line)
        return phrase_dict

    def _replace(text, dct):
        text1 = ""
        phrases = sorted(chain.from_iterable(dct[word] for word in text.split()
        if word in dct) ,reverse=1, key=len)
        mysetphrases = set(phrases)
        phrases = list(mysetphrases)
        for phr in phrases:
            text = text.replace(phr, phr.replace(" ", "_"))
        text1 =  "".join(text)
        return text1

    def _join_text(text):
        text = _replace(text, _get_phrases(mstr_list_clean))
        return text

    return _join_text(text)

@complete
def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]
    
    # Lemmatize all words in documents.
    #lemmatizer = WordNetLemmatizer()
    #docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
  
    return docs


def _calculate_lda(line):
    line = line.split()
    line_bow = id2word.doc2bow(line)
    doc_lda = ldamodel[line_bow]
    return([doc[0] for doc in doc_lda])


def _calculate_lda_p(line):
    line = line.split()
    line_bow = id2word.doc2bow(line)
    doc_lda = ldamodel[line_bow]
    return([(round(doc[1]*100,2)) for doc in doc_lda])


@complete
def cn_ci(text):
    out = ['vi'+str(index) for index, word in enumerate(mstr_list_clean_joined) if str(word) == str(text)]
    return out


@complete
def cn_ci1(text):
    text_l = text.split()
    out_l = []
    for id, txt in enumerate(text_l):
        out = ['vi'+str(index) for index, word in enumerate(mstr_list_clean) if str(word) == str(txt)]
        out_l.append(out)
    out_l = [item for sublist in out_l for item in sublist]
    unique_out_l = []
    for x in out_l:
      if(x not in unique_out_l):
        unique_out_l.append(x)
    return unique_out_l

def concept_word(text):
    text_l = text.split()
    out_l = []
    for id, txt in enumerate(text_l):
        out = [word for index, word in enumerate(mstr_list_clean) if str(word) == str(txt)]
        out_l.append(out)
    out_l = [item for sublist in out_l for item in sublist]
    unique_out_l = []
    for x in out_l:
      if(x not in unique_out_l):
        unique_out_l.append(x)
    return unique_out_l


def _calculate_tfidf(text_list):
    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(text_list)
    df = pd.DataFrame(tfidf.todense(), columns = tfidf_vect.get_feature_names())
    # df = pd.DataFrame(tfidf[0].T.todense(), index=tfidf_vect.get_feature_names_out(), columns=["TF-IDF"])
    return df

@complete
def _feat(txt):
    # print("****************************************************************************")
    response = vectorizer.transform([txt])
    feature_names = vectorizer.get_feature_names()
    arr = []
    for col in response.nonzero()[1]:
        if feature_names[col] in mstr_list_clean:
            f_n = ['vi'+str(index) for index, word in enumerate(mstr_list_clean_joined) if word in feature_names[col]]
            arr.append([f_n[0], response[0, col]])
        else:
            f_n = 'NA'
        # print(feature_names[col], response[0, col])
        # print(f_n, response[0, col])
        # print("=====")
    return arr


def _calculate_tfidf(text_list):
    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(text_list)
    df = pd.DataFrame(tfidf.todense(), columns = tfidf_vect.get_feature_names())
    # df = pd.DataFrame(tfidf[0].T.todense(), index=tfidf_vect.get_feature_names_out(), columns=["TF-IDF"])
    return df


# =============================================== Preprocessing ==============================================

MASTER_LIST_PATH = 'vocab_master_list.csv'
INTERMEDIATES_BASE_DIR = 'data_intermediates/'
TEST_CSV_PATH = 'test.csv'
TRIPLES_WITH_PROB = 'triples.csv'
V2_DF_PATH = f'{INTERMEDIATES_BASE_DIR}/v2_df.csv'
TW_DF1_PATH = f'{INTERMEDIATES_BASE_DIR}/tw_df1.csv'
ADF_PC_1_PATH = f'{INTERMEDIATES_BASE_DIR}/adf_pc_1.csv'
ADF_WITH_TFIDF_PATH = f'{INTERMEDIATES_BASE_DIR}/adf_with_tfidf.csv'


mstr_list_clean, mstr_list_clean_joined = load_words(MASTER_LIST_PATH)
pd.DataFrame(mstr_list_clean_joined).to_csv(V2_DF_PATH)
adf = load_test_csv(TEST_CSV_PATH)


adf['join_text'] = adf['text'].apply(lambda x: process_text(x))


text_list = adf['join_text'].to_list()
doc_tokens = docs_preprocessor(text_list)
dictionary = gensim.corpora.Dictionary(doc_tokens)
corpus = [dictionary.doc2bow(text) for text in doc_tokens]


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(corpus, num_topics=15, id2word = dictionary, passes=20, iterations=400, chunksize = 500, eval_every = None,random_state=0)
id2word = dictionary

topics = []
for topic_num in range(0,14):
    topic = ldamodel.show_topic(topic_num, topn=100)
    
    topic = [list(t) for t in topic]
    topic = [[t[0],t[1],topic_num] for t in topic]
    topics.append(topic)

topics = [item for items in topics for item in items]   
tw_df = pd.DataFrame(topics, columns =['feature', 'proba', 'topic_num'])


adf['l_text_topics'] = adf['join_text'].apply(lambda x: _calculate_lda(x))
adf['l_text_prob'] = adf['join_text'].apply(lambda x: _calculate_lda_p(x))


adf['concept_vocab_index'] = adf['join_text'].map(lambda s:cn_ci1(s))
adf['concept_vocab_word'] = adf['join_text'].map(lambda s:concept_word(s))


tw_df['cv_index'] = tw_df['feature'].map(lambda s:cn_ci(s))


tw_df1 = tw_df[tw_df['cv_index'].str.len() != 0]
tw_df1 = tw_df1.dropna()
tw_df1['cv_index_1'] = tw_df1['cv_index'].map(lambda s:s[0])

tw_df1.to_csv(TW_DF1_PATH)

corpus = adf['join_text']
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
adf['new'] = adf['join_text'].apply(lambda x: _feat(x))

adf['course_no'] = adf['course_name'].str.extract('(\d+)', expand=False)
adf['course_no'] = adf['course_no'].astype(int)
adf['prerequisite'] = adf.groupby('course_no').file_name.shift(1)
adf['composite'] = adf.groupby('course_no').file_name.shift(-1)

adf.to_csv(ADF_PC_1_PATH)

v2_df = pd.read_csv(V2_DF_PATH)

temptfidf = _calculate_tfidf(adf['join_text'])

cols = temptfidf.columns

cvw_tfidf_score = []
for i in range(len(temptfidf)):
  cvw = adf.loc[i]['concept_vocab_word']
  tfidf = temptfidf.loc[i]
  cvw_tfidf = []
  for word in cvw:
    for ind, col in enumerate(cols):
      if (word==col):
        cvw_tfidf.append(round(tfidf[ind],2))
  cvw_tfidf_score.append(cvw_tfidf)


adf['concept_vocab_word_tfidf'] = cvw_tfidf_score

adf.to_csv(ADF_WITH_TFIDF_PATH)

adf1 = adf[['file_name','l_text_topics']]
adf1 = adf1.dropna()
adf1 = adf1.melt('file_name')
adf1.rename(columns = {"file_name": "head"}, inplace = True) 
adf1['prob'] = adf[['l_text_prob']]
adf1 = adf1.explode(['value', 'prob'])
adf1['value'] = 'topic_' + adf1['value'].astype(str)


adf2 = adf[['file_name','concept_vocab_index']]
adf2 = adf2.dropna()
adf2 = adf2.melt('file_name')
adf2.rename(columns = {"file_name": "head"}, inplace = True) 
adf2['prob'] = adf['concept_vocab_word_tfidf']
# adf2['prob'] = adf2['prob'].apply(lambda x:ast.literal_eval(x) if x != [] else 0)
# adf2['value'] = adf2['value'].apply(lambda x:ast.literal_eval(x) if x != [] else 0)
print(len(adf2['prob'].iloc[0]), len(adf2['value'].iloc[0]))
adf2 = adf2.explode(['value','prob'])
adf2 = adf2.drop_duplicates(subset=['head','variable','value','prob'], keep='last')


adf['prerequisite'] = adf['prerequisite'].fillna('start')
adf['composite'] = adf['composite'].fillna('end')
adf3 = adf[['file_name','prerequisite']]
adf3 = adf3.dropna()
adf3 = adf3.melt('file_name')
adf3.rename(columns = {"file_name": "head"}, inplace = True) 
adf3 = adf3.drop_duplicates(subset=['head','variable','value'], keep='last')


adf4 = adf[['file_name']]
adf4['variable'] = 'level'
adf4['value'] = 'level_1'
adf4['value'] = adf4.apply(lambda row: 'level_1' if row['file_name'][6] == '1' else ('level_2' if row['file_name'][6] == '3' else 'level_3'), axis=1)
adf4.rename(columns = {"file_name": "head"}, inplace = True) 


#topics to vacab have to be done separately
# tw_df1 = pd.read_csv('tw_df1.csv')
adf5 = tw_df1[['topic_num','cv_index_1']]
adf5 = adf5.dropna()
adf5 = adf5.melt('topic_num')
adf5.rename(columns = {"topic_num": "head"}, inplace = True) 
adf5 = adf5.drop_duplicates(subset=['head','variable','value'], keep='last')
adf5['head'] = adf5['head'] + 1
adf5['head'] = 'topic_' + adf5['head'].astype(str)
adf5['variable'] = "concept_vocab_index"
adf5.head()

fdf = pd.concat([adf1,adf2,adf3,adf4,adf5])
fdf = fdf.fillna(1)

fdf.drop_duplicates()
cvi_triples = fdf[fdf['variable'] == 'concep_vocab_index']
l_text_triples = fdf[fdf['variable'] == 'l_text_topics']
prerequisite_triples = fdf[fdf['variable'] == 'prerequisite']
level_triples = fdf[fdf['variable'] == 'level']

cvi_triples['prob'] = MinMaxScaler().fit_transform(cvi_triples['prob'])
l_text_triples['prob'] = MinMaxScaler().fit_transform(l_text_triples['prob'])
prerequisite_triples['prob'] = MinMaxScaler().fit_transform(prerequisite_triples['prob'])
level_triples['prob'] = MinMaxScaler().fit_transform(level_triples['prob'])

fdf = pd.concat([cvi_triples, l_text_triples, prerequisite_triples, level_triples])

fdf.to_csv(TRIPLES_WITH_PROB)
print(f"GENERATED TRIPLES FILES AT -> {os.path.relpath(TRIPLES_WITH_PROB)}")


# ============================================================================================================