import pandas as pd
import numpy as np
from MF import MF
from metrics import ndcg
import implicit
from scipy import sparse
path = '~/PycharmProjects/VK/ratings.csv'
df = pd.read_csv(path,nrows = 100000)
#print(df)

movieID = np.array(df['movieId'])
rating = np.array(df['rating'])
userID = np.array(df['userId'])

Z = np.zeros((100,100))
counterID = 1 # -1
for i in range(len(userID)):
    userid = userID[i]
    movideid = movieID[i]
    if (movideid <=100 and userid <=100):
        Z[userid-1][movideid-1] = rating[i]



mf = MF(Z, K=2, alpha=0.1, beta=0.01, iterations=20)
mf.train()
t = mf.full_matrix()
ndcg1_mf = 0
ndcg10_mf = 0
for i in range(100):
    ndcg1_mf+= ndcg(t[i],1)
    ndcg10_mf+=ndcg(t[i],10)


model = implicit.als.AlternatingLeastSquares()
sZ = sparse.csc_matrix(Z)
sZi = sZ.T.tocsr()

model.fit(sZ)
ndcg1_als = 0
ndcg10_als = 0
#print(model.recommend(1,sZi,N=10))
for i in range(100):
    tmp = model.recommend(i,sZi)
    als_tmp = []
    for j in range(len(tmp)):
        als_tmp.append(tmp[j][1])
    ndcg1_als+=ndcg(als_tmp,1)
    ndcg10_als+=ndcg(als_tmp,10)
print("ndcg@1 for MF = {}, ndcg@10 ={}".format(ndcg1_mf,ndcg10_mf))
print("ndcg@1 for ALS = {},ndcg@10 = {}".format(ndcg1_als,ndcg10_als))
