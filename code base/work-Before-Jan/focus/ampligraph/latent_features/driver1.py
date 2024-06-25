import numpy as np
#from ampligraph.latent_features import TransE
#from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
#from .latent_features import TransE
from .models.TransH import TransH
#from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
model = TransH(batches_count=1, seed=555, epochs=20,
               k=10, loss='pairwise',
               loss_params={'margin':5})
X = np.array([['a', 'y', 'b'],
              ['b', 'y', 'a'],
              ['a', 'y', 'c'],
              ['c', 'y', 'a'],
              ['a', 'y', 'd'],
              ['c', 'y', 'd'],
              ['b', 'y', 'c'],
              ['f', 'y', 'e']])

# Numeric values below are associate to each triple in X.
# They can be any number and will be automatically
# normalised to the [0, 1] range, on a
# predicate-specific basis.
X_edge_values = np.array([5.34, -1.75, 0.33, 5.12,
                          np.nan, 3.17, 2.76, 0.41])

model.fit(X, focusE_numeric_edge_values=X_edge_values)
model.fit(X)
print("done")
#ranks = evaluate_performance(X, model=model, use_default_protocol=True, verbose=True)

# compute and print metrics:
#mrr = mrr_score(ranks)
#hits_10 = hits_at_n_score(ranks, n=10)
#print("MRR: %f, Hits@10: %f" % (mrr, hits_10))# -*- coding: utf-8 -*-

