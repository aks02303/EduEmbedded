import numpy as np
import pandas as pd
#from ampligraph.latent_features import TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, mr_score, hits_at_n_score
#from .latent_features import TransE
from .models.TransH3 import TransH
#from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
import os
# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
import os
# X = pd.read_csv("/home/sk/IIITB/sem2/EduEmbed/code/data/triples_with_prob_tfidf.csv")
relpath= os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
X = pd.read_csv(relpath+'/data/triples.csv')


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

# scaler = MinMaxScaler()
# l_text_topics['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(l_text_topics['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# concept_vocab_index['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(concept_vocab_index['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# prerequisite['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(prerequisite['weights'])), columns = ['weights'])

# scaler = MinMaxScaler()
# level['weights'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(level['weights'])), columns = ['weights'])

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

# data.to_csv('/home/sk/IIITB/sem2/EduEmbed/EduEmbedd/embeddings_final/april_7/transH/train_triples.csv')

hyperparameter_result=pd.DataFrame(columns=['model_name', 'epochs', 'batches_count', 'k', 'structural_wt', 'lr','start_loss','end_loss', 'mrr' ,'mr', 'hits_10', 'hits_5', 'hits_3', 'hits_1', 'losses_list'])


def train_model(epochs, batches_count, k, structural_wt, lr, entities=entities, relations=relations):
    model = TransH(batches_count=batches_count, seed=555, epochs=epochs, k=k, loss='pairwise', loss_params={'margin':5}, verbose = True, embedding_model_params={'structural_wt':structural_wt}, optimizer_params = {'lr':lr})
    losses_list = model.fit(train_triples, focusE_numeric_edge_values=train_weights)

    path = '/home/sheetal/iiitb/sem2/wsl/EduEmbedd-main/code/embeddings_final/transH/'
    dirname = "transH_"+str(epochs)+"_"+str(batches_count)+"_"+str(k)+"_"+str(structural_wt)+"_"+str(lr)
    path = path+dirname
    os.mkdir(path)

# Numeric values below are associate to each triple in X.
# They can be any number and will be automatically
# normalised to the [0, 1] range, on a
# predicate-specific basis.
#model.fit(X)

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

    # x = list(range(1, epochs+1))  # X-axis points
    # y = losses_list  # Y-axis points
    
    # plt.plot(x, y)  # Plot the chart
    # plt.show()  # display


# train_model(1500, 5, 40, 0.8, 0.05)
# train_model(5000, 5, 50, 0.8, 0.005)
#train_model(100, 5, 45, 0.8, 0.005)
lrs=[0.1,0.01]
embed_size=[40,50]
epochs=[50,100]

for lr in lrs:
    for es in embed_size:
        for epoch in epochs:
            train_model(epoch, 5, es, 0.1, lr)

#hyperparameter_result.to_csv('/home/sheetal/iiitb/sem2/wsl/EduEmbedd-main/code/embeddings_final/transH_hp_result_old.csv')
hyperparameter_result.to_csv(relpath+'/embeddings_final/output/transH_hyperparam_result.csv')

# plt.show()



#ranks = evaluate_performance(X, model=model, use_default_protocol=True, verbose=True)

# compute and print metrics:
#mrr = mrr_score(ranks)
#hits_10 = hits_at_n_score(ranks, n=10)
#print("MRR: %f, Hits@10: %f" % (mrr, hits_10))# -*- coding: utf-8 -*-
