import os

import numpy as np

import pandas as pd

from surprise import Reader, Dataset

import scipy.io

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
ds_dir = '../input/pda2019'
train = pd.read_csv(os.path.join(ds_dir,"train-PDA2019.csv"))

test = pd.read_csv(os.path.join(ds_dir,"test-PDA2019.csv"))
train
itemIDs = train.itemID.unique()
test
train.drop('timeStamp',inplace=True,axis=1)
userIDs = train.userID.unique()
watchedList = train.groupby('userID')['itemID'].apply(list)
reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(train, reader)

trainset = data.build_full_trainset()
def predict(user):

    pred = []

    for x in itemIDs:

        if x in watchedList[user.userID]:

            continue

        pred.append((x,model.predict(user.userID,x).est))

    pred = sorted(pred, key = lambda x: x[1], reverse=True)[:10]

    pred = [i[0] for i in pred]

    pred = ' '.join(map(str, pred)) 

    return pred
def get_top10(row):

    if row.userID in userIDs:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = pred[~pred.itemID.isin()]

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred

    else:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred

model = SVD(verbose=1,random_state=0).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SVD.csv',index=False)

model = SVDpp(verbose=1,random_state=0).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SVDpp.csv',index=False)

model = SlopeOne().fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SlopeOne.csv',index=False)

model = NMF(verbose=1,random_state=0).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_NMF.csv',index=False)

model = NormalPredictor().fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_NormalPredictor.csv',index=False)

model = KNNBaseline(verbose=1,bsl_options={'method':'als'}).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBaseline_ALS.csv',index=False)

model = KNNBaseline(verbose=1,bsl_options={'method':'sgd'}).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBaseline_SGD.csv',index=False)

model = KNNBasic(verbose=1).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBasic.csv',index=False)

model = KNNWithMeans(verbose=1).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNWithMeans.csv',index=False)

model = KNNWithZScore(verbose=1).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNWithZScore.csv',index=False)

model = BaselineOnly(verbose=1,bsl_options={'method':'als'}).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_BaselineOnly_ALS.csv',index=False)

model = BaselineOnly(verbose=1,bsl_options={'method':'sgd'}).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_BaselineOnly_SGD.csv',index=False)

model = CoClustering(verbose=1,random_state=0).fit(trainset)

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_CoClustering.csv',index=False)