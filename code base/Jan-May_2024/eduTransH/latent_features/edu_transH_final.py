import subprocess
import pandas as pd 
import numpy as np 
import sys
from pyvis.network import Network
import random
from collections import defaultdict
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D 
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re
import sys
import os
import string
import nltk
import argparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path
import numpy as np
import pandas as pd
#from ampligraph.latent_features import TransE
import os
import subprocess
from ampligraph.evaluation import evaluate_performance, mrr_score, mr_score, hits_at_n_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from .latent_features import TransE
from .models.TransH3 import TransH
#from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
import os
# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

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
def load_data_files(path_to_data_files):
    results = defaultdict(list)
    for file in Path(path_to_data_files).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            if(file.name[-3:]!="txt"):
                print(file.name)
                continue

            results["file_name"].append(file.name.split(".")[0])
            results["text"].append(file_open.read())
    adf = pd.DataFrame(results)
    return adf

@complete
def extract_details(adf):
    adf['course_name'] = adf.file_name.str.split('_').str[0]
    adf['temp'] = adf.file_name.str.split('_').str[1]
    adf['week'] = adf.temp.str.split('-').str[0]
    adf['section'] = adf.temp.str.split('-').str[1]
    adf['lesson'] = adf.temp.str.split('-').str[2]
    adf['course_title'] = adf.file_name.str.split('_').str[2]
    adf.dropna(inplace=True)

    adf['week_no'] = adf['week'].str.extract('(\d+)', expand=False)
    adf['week_no'] = adf['week_no'].astype(int)
    adf['section_no'] = adf['section'].str.extract('(\d+)', expand=False)
    adf['section_no'] = adf['section_no'].astype(int)
    adf['lesson_no'] = adf['lesson'].str.extract('(\d+)', expand=False)
    adf = adf.dropna()
    adf['lesson_no'] = adf['lesson_no'].astype(int)
    return adf

@complete
def clean_text_column(adf):
    adf['text'] = adf['text'].apply(lambda x: re.sub("\\n", " ",x))
    adf['text'] = adf['text'].apply(lambda x: re.sub(r'\s+',' ', x))
    adf['text'] = adf['text'].apply(lambda x: x.lower())
    return adf

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

@complete
def stem_and_stopword_removal(adf):

    # remove punctuation
    adf['text'] = adf['text'].apply(lambda x: remove_punct(x))

    nltk.download('stopwords')
    stopword = nltk.corpus.stopwords.words('english')
    adf['text'] = adf['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword)]))
    # stopwords 

    ps = nltk.PorterStemmer()

    adf['text1'] = adf['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    return adf

def _calculate_nmf(text_list) :
    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(text_list)
    # tfidf_feature_names = tfidf_vect.get_feature_names()
    # print(tfidf)
    nmf = NMF(n_components=15,random_state=1, alpha=.1, l1_ratio=.7, init='nndsvd').fit(tfidf)
    nmf_trans = nmf.transform(tfidf)
    # print(nmf_trans)
    predicted_topics = [np.argsort(each)[::-1][0:5] for each in nmf_trans]
    return predicted_topics

def calc_text_topics(adf):
    adf['text_topics'] = _calculate_nmf(adf['text1'])
    return adf

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

def _calculate_lda_p_topic(line):
    line = line.split()
    line_bow = id2word.doc2bow(line)
    doc_lda = ldamodel[line_bow]
    return([[doc[0], round(doc[1]*100,2)] for doc in doc_lda])



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

# =============================================== Arguments Parsing ==============================================

parser = argparse.ArgumentParser(description='Preprocessing script')

parser.add_argument('--debug', action='store_true', help='Enable Debug mode')
parser.add_argument('command', choices=['tsne'], help='The command to execute')
args = parser.parse_args()


# =============================================== Preprocessing ==============================================


DEBUG_MODE = args.debug     # set to True to generate intermediate files (in data_intermediates) for debugging
if DEBUG_MODE: print("DEBUG MODE ENABLED")

PREREQ_LEVEL = False        # set to True to generate prerequisite and level triples



MASTER_DATA_PATH = input("Enter the location of input data folder")
MASTER_LIST_PATH = input("Enter locatoin of master_vocab_list")



# INTERMEDIATES_BASE_DIR = 'debug_output'                # directory to store debug files (if debug option selected)
TRIPLES_WITH_PROB = input('Enter output folder location')
INTERMEDIATES_BASE_DIR=TRIPLES_WITH_PROB
TEST_CSV_PATH = f'{INTERMEDIATES_BASE_DIR}/test.csv'
V2_DF_PATH = f'{INTERMEDIATES_BASE_DIR}/v2_df.csv'
TW_DF1_PATH = f'{INTERMEDIATES_BASE_DIR}/tw_df1.csv'
ADF_PC_1_PATH = f'{INTERMEDIATES_BASE_DIR}/adf_pc_1.csv'
ADF_WITH_TFIDF_PATH = f'{INTERMEDIATES_BASE_DIR}/adf_with_tfidf.csv'



# extract features from the data files
adf = load_data_files(MASTER_DATA_PATH)
if PREREQ_LEVEL: adf = extract_details(adf)
adf = clean_text_column(adf)
adf = stem_and_stopword_removal(adf)
if PREREQ_LEVEL: adf = calc_text_topics(adf)

if DEBUG_MODE: adf.to_csv(TEST_CSV_PATH)


mstr_list_clean, mstr_list_clean_joined = load_words(MASTER_LIST_PATH)
v2_df = pd.DataFrame(mstr_list_clean_joined)
if DEBUG_MODE: v2_df.to_csv(V2_DF_PATH)
# adf = load_test_csv(TEST_CSV_PATH)


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


# adf['l_text_topics'] = adf['join_text'].map(_calculate_lda)
# adf['l_text_prob'] = adf['join_text'].apply(lambda x: _calculate_lda_p(x))

# above lines replaced with the following due to unexpected behaviour
adf['temp_text'] = adf['join_text'].apply(lambda x: _calculate_lda_p_topic(x))
adf['l_text_topics'] = adf['temp_text'].apply(lambda x: [item[0] for item in x])
adf['l_text_prob'] = adf['temp_text'].apply(lambda x: [item[1] for item in x])
adf.drop(columns=['temp_text'], inplace=True)



adf['concept_vocab_index'] = adf['join_text'].map(lambda s:cn_ci1(s))
adf['concept_vocab_word'] = adf['join_text'].map(lambda s:concept_word(s))


tw_df['cv_index'] = tw_df['feature'].map(lambda s:cn_ci(s))


tw_df1 = tw_df[tw_df['cv_index'].str.len() != 0]
tw_df1 = tw_df1.dropna()
tw_df1['cv_index_1'] = tw_df1['cv_index'].map(lambda s:s[0])

if DEBUG_MODE: tw_df1.to_csv(TW_DF1_PATH)

corpus = adf['join_text']
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
adf['new'] = adf['join_text'].apply(lambda x: _feat(x))

if PREREQ_LEVEL:
    adf['course_no'] = adf['course_name'].str.extract('(\d+)', expand=False)
    adf['course_no'] = adf['course_no'].astype(int)
    adf['prerequisite'] = adf.groupby('course_no').file_name.shift(1)
    adf['composite'] = adf.groupby('course_no').file_name.shift(-1)


if DEBUG_MODE: adf.to_csv(ADF_PC_1_PATH)

temptfidf = _calculate_tfidf(adf['join_text'])

cols = temptfidf.columns

cvw_tfidf_score = []
for i in range(len(temptfidf)):
  cvw = adf.iloc[i]['concept_vocab_word']
  tfidf = temptfidf.iloc[i]
  cvw_tfidf = []
  for word in cvw:
    for ind, col in enumerate(cols):
      if (word==col):
        cvw_tfidf.append(round(tfidf[ind],2))
  cvw_tfidf_score.append(cvw_tfidf)


adf['concept_vocab_word_tfidf'] = cvw_tfidf_score

if DEBUG_MODE: adf.to_csv(ADF_WITH_TFIDF_PATH)

adf1 = adf[['file_name','l_text_topics']]
adf1 = adf1.dropna()
adf1 = adf1.melt('file_name')
adf1.rename(columns = {"file_name": "head"}, inplace = True) 
adf1['prob'] = adf['l_text_prob'].to_list()
adf1 = adf1.explode(['value', 'prob'])
adf1['value'] = 'topic_' + adf1['value'].astype(str)


adf2 = adf[['file_name','concept_vocab_index']]
adf2 = adf2.dropna()
adf2 = adf2.melt('file_name')
adf2.rename(columns = {"file_name": "head"}, inplace = True) 
adf2['prob'] = adf['concept_vocab_word_tfidf'].to_list()
# adf2['prob'] = adf2['prob'].apply(lambda x:ast.literal_eval(x) if x != [] else 0)
# adf2['value'] = adf2['value'].apply(lambda x:ast.literal_eval(x) if x != [] else 0)
# print(len(adf2['prob'].iloc[0]), len(adf2['value'].iloc[0]))
adf2 = adf2.explode(['value','prob'])
adf2 = adf2.drop_duplicates(subset=['head','variable','value','prob'], keep='last')

if PREREQ_LEVEL:
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

if PREREQ_LEVEL: fdf = pd.concat([adf1,adf2,adf3,adf4,adf5])
else: fdf = pd.concat([adf1,adf2,adf5])

fdf = fdf.fillna(1)

fdf.drop_duplicates()
cvi_triples = fdf[fdf['variable'] == 'concept_vocab_index']
l_text_triples = fdf[fdf['variable'] == 'l_text_topics']
if PREREQ_LEVEL: prerequisite_triples = fdf[fdf['variable'] == 'prerequisite']
if PREREQ_LEVEL: level_triples = fdf[fdf['variable'] == 'level']


# print(fdf.shape)
cvi_triples['prob'] = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(cvi_triples['prob'])), columns = ['prob'])
l_text_triples['prob'] = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(l_text_triples['prob'])), columns = ['prob'])
if PREREQ_LEVEL: prerequisite_triples['prob'] = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(prerequisite_triples['prob'])), columns = ['prob'])
if PREREQ_LEVEL: level_triples['prob'] = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(level_triples['prob'])), columns = ['prob'])

if PREREQ_LEVEL: fdf = pd.concat([cvi_triples, l_text_triples, prerequisite_triples, level_triples])
else: fdf = pd.concat([cvi_triples, l_text_triples])

fdf.to_csv(f'{TRIPLES_WITH_PROB}/triples.csv')
print(f"GENERATED TRIPLES FILES AT -> {os.path.relpath(TRIPLES_WITH_PROB)}")





# TRANSH_OUTPUT_PATH = r'D:\PE\EduEmbedd\code base\Jan-May_2024\eduTransH\transH_without_weights'
TRANSH_OUTPUT_PATH = TRIPLES_WITH_PROB
TRANSH_OUTPUT_PATH=TRANSH_OUTPUT_PATH+"/transH_without_weight"
# print("here 1")
if os.path.isdir(TRANSH_OUTPUT_PATH):
    shutil.rmtree(TRANSH_OUTPUT_PATH)
os.mkdir(TRANSH_OUTPUT_PATH)

relpath =TRANSH_OUTPUT_PATH
# TRIPLES_WITH_PROB=f'{TRIPLES_WITH_PROB}/triples.csv'
X = pd.read_csv(f'{TRIPLES_WITH_PROB}/triples.csv')


X.rename(columns = {'prob':'weights'}, inplace = True)
# X = X.sample(frac=1)


X['variable'].unique()
index = X[ (X['head'].astype(str).str.match("topic+"))].index
X.drop(index, inplace=True)



l_text_topics = X[X['variable']=='l_text_topics']
concept_vocab_index = X[X['variable']=='concept_vocab_index']
prerequisite = X[X['variable']=='prerequisite']
level = X[X['variable']=='level']


l_text_topics = l_text_topics.sample(frac=1)
concept_vocab_index = concept_vocab_index.sample(frac=1)
prerequisite = prerequisite.sample(frac=1)
level = level.sample(frac=1)

scaler = MinMaxScaler()
l_text_topics['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(l_text_topics['weights'])), columns = ['weights'])

scaler = MinMaxScaler()
concept_vocab_index['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(concept_vocab_index['weights'])), columns = ['weights'])

fractions = np.array([0.7, 0.15, 0.15])
# fractions = np.array([0.8, 0.2])

ltt_train, ltt_val, ltt_test = np.array_split(l_text_topics, (fractions[:-1].cumsum() * len(l_text_topics)).astype(int))

cvi_train, cvi_val, cvi_test = np.array_split(concept_vocab_index, (fractions[:-1].cumsum() * len(concept_vocab_index)).astype(int))

pre_train, pre_val, pre_test = np.array_split(prerequisite, (fractions[:-1].cumsum() * len(prerequisite)).astype(int))

lvl_train, lvl_val, lvl_test = np.array_split(level, (fractions[:-1].cumsum() * len(level)).astype(int))


train = pd.concat([ltt_train, cvi_train, pre_train, lvl_train])
val = pd.concat([ltt_val, cvi_val, pre_val, lvl_val])
test = pd.concat([ltt_test, cvi_test, pre_test, lvl_test])
train = train.sample(frac=1)
val = val.sample(frac=1)
test = test.sample(frac=1)

# train, val, test = np.array_split(X, (fractions[:-1].cumsum() * len(X)).astype(int))

train_triples = train[['head', 'variable', 'value']].to_numpy()
train_weights = train[['weights']].to_numpy()

val_triples = val[['head', 'variable', 'value']].to_numpy()
val_weights = val[['weights']].to_numpy()

test_triples = test[['head', 'variable', 'value']].to_numpy()
test_weights = test[['weights']].to_numpy()

data = pd.DataFrame(train_triples, columns=['head', 'variable', 'value'])
data['head'].unique()
data['value'].unique()
data['variable'].unique()

data['head'].nunique()
data['value'].nunique()
data['variable'].nunique()

entities = data['head'].unique()
entities = np.append(entities, data['value'].unique())
# print(entities.shape)
entities = np.unique(entities)
# print(entities.shape)


relations = data['variable'].unique()
relations.shape


hyperparameter_result=pd.DataFrame(columns=['model_name', 'epochs', 'batches_count', 'k', 'structural_wt', 'lr','start_loss','end_loss', 'mrr' ,'mr', 'hits_10', 'hits_5', 'hits_3', 'hits_1', 'losses_list'])


def train_model(epochs, batches_count, k, structural_wt, lr, entities=entities, relations=relations):
    model = TransH(batches_count=batches_count, seed=555, epochs=epochs, k=k, loss='pairwise', loss_params={'margin':5}, verbose = True, embedding_model_params={'structural_wt':structural_wt}, optimizer_params = {'lr':lr})
    # losses_list = model.fit(train_triples, focusE_numeric_edge_values=train_weights) # for with weights
    losses_list = model.fit(train_triples) # for without weigths


    path = TRANSH_OUTPUT_PATH
    dirname = "/transH_"+str(epochs)+"_"+str(batches_count)+"_"+str(k)+"_"+str(structural_wt)+"_"+str(lr)
    path = path+dirname
    os.mkdir(path)

    data.to_csv(path+'/train_triples.csv')

    entity_embeddings = model.get_embeddings(entities, embedding_type='entity')
    relation_embeddings = model.get_embeddings(relations, embedding_type='relation')

    entities = pd.DataFrame(entities)
    entity_embeddings = pd.DataFrame(entity_embeddings)
    entities.to_csv(path + '/entities.csv', index=False, header=False)
    entity_embeddings.to_csv(path + '/entity_embeddings.csv', index=False, header=False)

    relations = pd.DataFrame(relations)
    relation_embeddings = pd.DataFrame(relation_embeddings)
    relations.to_csv(path + '/relations.csv', index=False, header=False)
    relation_embeddings.to_csv(path + '/relation_embeddings.csv', index=False, header=False)


    filter = np.concatenate((train_triples, val_triples, test_triples))
# filter = np.concatenate((train_triples, test_triples))

    ranks = evaluate_performance(val_triples, model = model, filter_triples = filter, use_default_protocol=True, verbose=True)
    mrr = mrr_score(ranks)
    mr = mr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    # print("MRR: %f, MR: %f, Hits@10: %f" % (mrr, mr, hits_10))
    hits_5 = hits_at_n_score(ranks, n=5)
    # print("Hits@5: %.2f" % (hits_5))
    hits_3 = hits_at_n_score(ranks, n=3)
    # print("Hits@3: %.2f" % (hits_3))
    hits_1 = hits_at_n_score(ranks, n=1)
    # print("Hits@1: %.2f" % (hits_1))
    # print("done")

    start_loss = losses_list[0]
    end_loss = losses_list[-1]
    row= ['transH', epochs, batches_count, k, structural_wt, lr,start_loss, end_loss, mrr ,mr,hits_10,hits_5,hits_3,hits_1, losses_list]
    print(row)
    hyperparameter_result.loc[len(hyperparameter_result)]=row

lrs=[0.1,0.01]
embed_size=[40, 50]
epochs=[50, 100]

for lr in lrs:
    for es in embed_size:
        for epoch in epochs:
            train_model(epoch, 5, es, 0.1, lr)

#hyperparameter_result.to_csv('/home/sheetal/iiitb/sem2/wsl/EduEmbedd-main/code/embeddings_final/transH_hp_result_old.csv')
hyperparameter_result.to_csv(relpath+'/transH_without_weights_hyperparam_result.csv')



def generate_graph_with_neighbors(nodes, data):
    # Convert data to DataFrame
    data_df = pd.DataFrame(data, columns=['head', 'variable', 'value'])
    
    # Filter data to include only the specified nodes and their neighbors
    node_data = data_df[data_df['head'].isin(nodes) | data_df['value'].isin(nodes)]
    neighbor_nodes = set(node_data['head']).union(set(node_data['value']))

    # Create a Pyvis Network instance
    net = Network(notebook=True, height='100vh')  # Set graph height to 100vh

    # Add nodes with random colors
    for node in neighbor_nodes:
        node_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate a random hex color code
        net.add_node(node, color=node_color)

    # Add edges with relation information
    for _, row in node_data.iterrows():
        net.add_edge(row['head'], row['value'], title=row['variable'], label='', hover_title=row['variable'])

    # Configure physics and interaction options
    net.options.physics.enabled = True
    net.options.physics.barnesHut = {"gravitationalConstant": -2000, "springLength": 150, "springConstant": 0.03}
    net.options.physics.maxVelocity = 50
    net.options.physics.minVelocity = 0.1
    net.options.physics.solver = 'forceAtlas2Based'
    
    # Configure interaction options
    net.options.interaction.hover = True
    net.options.interaction.zoomView = True
    net.options.interaction.dragNodes = True
    net.options.interaction.dragView = True
    net.options.interaction.showNodeNamesOnHover = True
    net.options.interaction.tooltipDelay = 200  # Adjust tooltip delay
    
    return net # Show node names on hover only when zoomed in

# Function to take a comma-separated list of nodes as input
def get_node_list():
    nodes = input("Enter a comma-separated list of nodes: ").strip().split(',')
    return [node.strip() for node in nodes]

# Load the data
TRAIN_TRIPLES_FILE = TRIPLES_WITH_PROB+"/triples.csv"
OUTPUT_GRAPH_FILE = TRANSH_OUTPUT_PATH
try:
    data = pd.read_csv(TRAIN_TRIPLES_FILE)
except Exception as e:
    print("An error occurred while loading the data:", e)
    raise

# Get the choice from the user
print("Choose an option:")
print("1. Visualize the whole graph")
print("2. Visualize specific nodes")
choice = input("Enter your choice: ")

if choice == '1':
    # Extract all unique nodes from both 'head' and 'value' columns
    all_nodes = set(data['head']).union(set(data['value']))
    print(len(all_nodes))
    # Generate the graph with all nodes
    net = generate_graph_with_neighbors(all_nodes, data)
elif choice == '2':
    # Get the list of nodes from the user
    target_nodes = get_node_list()
    # Generate the graph with only the specified nodes and their neighbors
    net = generate_graph_with_neighbors(target_nodes, data)
else:
    print("Invalid choice.")
    exit()

# Show the visualization
net.show(f'{OUTPUT_GRAPH_FILE}/graph.html')

def plot_tsne_3d(entities_csv, embeddings_csv, output_path):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Assign colors based on prefixes
    colors = ['red' if entity.startswith('Course') else 'green' if entity.startswith('vi') else 'blue' if entity.startswith ('topic') else 'black' for entity in entity_df['entity']]

    # Perform t-SNE with 3 dimensions
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_tsne = tsne.fit_transform(embedding_df.iloc[:, :-1])  # Exclude the 'entity' column from embeddings

    # Plot the t-SNE visualization with different colors in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Scatter plot in 3D
    ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2], c=colors, alpha=0.5)

    # Create custom legend
    ax.scatter([], [], [], color='red', label='Courses')
    ax.scatter([], [], [], color='green', label='Vocabulary')
    ax.scatter([], [], [], color='blue', label='Topics')

    # Set labels and legend
    ax.set_title('t-SNE Visualization for All Entities (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.legend()

    plt.savefig(output_path + '/tsne_3d.png')
    plt.show()

def plot_tsne_2d(entities_csv, embeddings_csv, output_path):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Assign colors based on prefixes
    colors = ['red' if entity.startswith('Course') else 'green' if entity.startswith('vi') else 'blue' if entity.startswith('topic') else 'white' for entity in entity_df['entity']]

    # Perform t-SNE with 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embedding_df.iloc[:, :-1])  # Exclude the 'entity' column from embeddings

    # Plot the t-SNE visualization with different colors
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], color=colors, alpha=0.5)

    # Create custom legend
    plt.scatter([], [], color='red', label='Courses')
    plt.scatter([], [], color='green', label='Vocabulary')
    plt.scatter([], [], color='blue', label='Topics')

    plt.title('t-SNE Visualization for All Entities (2D)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()

    plt.savefig(output_path + '/tsne_2d.png')
    plt.show()

# Example usage:

if(args.command=="tsne"):
    plot_tsne_2d(f'{TRANSH_OUTPUT_PATH}/transH_50_5_40_0.1_0.1/entities.csv',
             f'{TRANSH_OUTPUT_PATH}/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
             f'{TRANSH_OUTPUT_PATH}')
    # plot_tsne_3d(f'{TRANSH_OUTPUT_PATH}/transH_50_5_40_0.1_0.1/entities.csv',
    #          f'{TRANSH_OUTPUT_PATH}/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
    #          f'{TRANSH_OUTPUT_PATH}')


