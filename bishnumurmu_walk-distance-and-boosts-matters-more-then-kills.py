# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.corr()
import matplotlib.pyplot as plt
plt.scatter(x ='winPlacePerc',y = 'walkDistance',data = train)
plt.xlabel('winPlacePerc')
plt.ylabel('walkDistance')
plt.scatter(x ='winPlacePerc',y = 'boosts',data = train)
plt.xlabel('winPlacePerc')
plt.ylabel('boosts')
plt.scatter(x ='winPlacePerc',y = 'kills',data = train)
plt.xlabel('winPlacePerc')
plt.ylabel('kills')
#kills doesn't matters much, but woh is this guy with 60 kills