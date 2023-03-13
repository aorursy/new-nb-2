# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import sys

import operator

import numpy as np

import pandas as pd

from scipy import sparse

import xgboost as xgb

from sklearn import model_selection, preprocessing, ensemble

from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
data_path = "../input/"

train_file = data_path + "train.json"

test_file = data_path + "test.json"

train_df = pd.read_json(train_file)

test_df = pd.read_json(test_file)

print(train_df.shape)

print(test_df.shape)
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]



# count of photos #

train_df["num_photos"] = train_df["photos"].apply(len)

test_df["num_photos"] = test_df["photos"].apply(len)



# count of "features" #

train_df["num_features"] = train_df["features"].apply(len)

test_df["num_features"] = test_df["features"].apply(len)



# count of words present in description column #

train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))

test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))



# convert the created column to datetime object so as to extract more features 

train_df["created"] = pd.to_datetime(train_df["created"])

test_df["created"] = pd.to_datetime(test_df["created"])



# Let us extract some features like year, month, day, hour from date columns #

train_df["created_year"] = train_df["created"].dt.year

test_df["created_year"] = test_df["created"].dt.year

train_df["created_month"] = train_df["created"].dt.month

test_df["created_month"] = test_df["created"].dt.month

train_df["created_day"] = train_df["created"].dt.day

test_df["created_day"] = test_df["created"].dt.day

train_df["created_hour"] = train_df["created"].dt.hour

test_df["created_hour"] = test_df["created"].dt.hour



# adding all these new features to use list #

features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])
categorical = ["display_address", "manager_id", "building_id", "street_address"]

for f in categorical:

        if train_df[f].dtype=='object':

            #print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train_df[f].values) + list(test_df[f].values))

            train_df[f] = lbl.transform(list(train_df[f].values))

            test_df[f] = lbl.transform(list(test_df[f].values))

            features_to_use.append(f)
#We have features column which is a list of string values. 

#So we can first combine all the strings together to get a single string and then apply count vectorizer on top of it.

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))

test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))

print(train_df["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(train_df["features"])

te_sparse = tfidf.transform(test_df["features"])
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()

test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()



target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, test_X.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

#clf.fit(train_X, train_y)

scores = cross_val_score(clf, train_X.toarray(), train_y, cv=3,scoring='neg_log_loss')

print (scores)
clf.fit(train_X.toarray(), train_y)

submission = clf.predict_proba(test_X.toarray())
out_df = pd.DataFrame(submission)

out_df.columns = ["high", "medium", "low"]

out_df["listing_id"] = test_df.listing_id.values

out_df.to_csv("xgb_starter2.csv", index=False)