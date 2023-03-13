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
import matplotlib.pyplot as plt

import seaborn as sns

spray = pd.read_csv('../input/spray.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

weather = pd.read_csv('../input/weather.csv')
spray.info()
spray.head()
train.info()
train.head()
weather.info()
pd.options.display.max_columns=20

weather.head()