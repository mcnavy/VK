import numpy as np
def dcg(r,k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0]+np.sum(r[1:]/np.log2(np.arange(2,r.size+1)))
    return 0
def ndcg(r,k):
    dcg_ideal = dcg(sorted(r,reverse = True),k)
    if not dcg_ideal:
        return 0
    return dcg(r,k)/dcg_ideal
