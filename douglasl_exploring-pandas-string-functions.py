import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from textblob import TextBlob

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Create pools

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# months = [str(i).zfill(2) for i in range(1, 13)]

days = [str(i).zfill(2) for i in range(1, 29)]

years = [str(i) for i in range(1992, 2016)]
# Randomly generate from pools

rand_months = np.random.choice(months, size=10000)

rand_days = np.random.choice(days, size=10000)

rand_years = np.random.choice(years, size=10000)
# Place into df

rand_dates.head()
rand_dates.loc[23, 'mo'] = ''
rand_dates.loc[23, 'mo'] = np.NaN
train = pd.read_csv('../input/train.csv', nrows=10000)

train2 = train.copy()

train['first'] = train['question_text'].str[0]

train['count_e'] = train['question_text'].str.count('e')

train['cap'] = train['question_text'].str.capitalize()

# Just the individual values added together
def extract_text_features(x):

    return x[0], x.count('e'), x.capitalize()

a,b,c = [], [], []

for s in train['question_text']:

    a.append(s[0]), b.append(s.count('e')), c.append(s.capitalize())

train['first'] = a

train['count_e'] = b

train['cap'] = c

# assigning to new column takes about the same time in either method
# bonus - getting memory of your array

train['question_text'].values.nbytes

train2['num_chars'] = train2['question_text'].str.len()

train2['is_titlecase'] = train2['question_text'].str.istitle().astype('int')

train2['has_*'] = train2['question_text'].str.contains(r'[A-Za-z]\*.|.\*[A-Za-z]', regex=True).astype('int')

def srs_funcs(srs):

    a = len(srs)

    b = int(srs.istitle())

    c = int(bool(re.search(r'[A-Za-z]\*.|.\*[A-Za-z]', srs)))

    return a, b, c

# would have expected this to be faster than creating three new columns individually but maybe the type conversion calls slowed it down
def srs_funcs2(srs):

    a = len(srs)

    b = int(srs.istitle())

    c = int(bool(re.search(r'[A-Za-z]\*.|.\*[A-Za-z]', srs)))

    return pd.Series([a, b, c])

# calling pd.series each time through loop kills performance
def textblob_methods(blob):

    '''Access Textblob methods and returns as tuple

    '''

    # convert to python list of tokens

    return blob.polarity, blob.subjectivity, int(blob.ends_with('?'))
train3 = pd.read_csv('../input/train.csv', nrows=10000)

train3.head()
# Convert  - any ways to make this faster? 

zsamp = train3.loc[5006]['blobs']

a, b, c = [], [], []

for s in train3['blobs']:

    a.append(s.polarity), b.append(s.subjectivity), c.append(int(s[-1] in '?'))

train3['polarity'], train3['subjectivity'], train3['ends_with_?'] = a, b, c

# Doing it separately - takes longer

train3['polarity'] = train3['blobs'].apply(lambda x: x.polarity)

train3['subjectivity'] = train3['blobs'].apply(lambda x: x.subjectivity)

train3['ends_with_?'] = train3['blobs'].apply(lambda x: x.endswith('?'))
def textblob_methods2(blob):

    '''Access Textblob methods and returns as tuple

    '''

    # convert to python list of tokens

    return blob.polarity, blob.subjectivity

a, b = [], []

for s in train3['blobs']:

    a.append(s.polarity), b.append(s.subjectivity)

train3['polarity'], train3['subjectivity'] = a, b