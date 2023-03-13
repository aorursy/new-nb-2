# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000000)
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[h]')
train_bookings = train1[train1['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train1[train1['is_booking'] == 0].drop('is_booking', axis=1)
train_clicks.shape
train_bookings.shape
train_bookingszna = train_bookings.fillna(0)
train_clickszna = train_clicks.fillna(0)
train_bookings_mobile = train_bookingszna[train_bookingszna['is_mobile'] == 1]
train_clicks_mobile = train_clickszna[train_clickszna['is_mobile'] == 1]
train_bookings_nonmobile = train_bookings[train_bookings['is_mobile'] == 0]
train_clicks_nonmobile = train_clicks[train_clicks['is_mobile'] == 0]
train_bookingszna_all = train_bookingszna['stay_span']
train_bookingszna_all.shape
bookedall_bookings_mob, countall = np.unique(train_bookingszna_all, return_counts=True)
np.median(countall)
np.sort(countall)
cb_with_mobile = booked_bookings_mob[count >= 211]
cb_with_mobile
train_bookingszna_mobile = train_bookingszna[train_bookingszna['stay_span'].isin(cb_with_mobile)]

train_bookings_mobile.shape
train_bookings_mobile.ix[(train_bookings_mobile['stay_span'] <= -1), 'stay_span'] = 0
f, ax = plt.subplots(figsize=(15, 25))
sns.countplot(y='stay_span', data=train_bookings_mobile)
sns.plt.title('Difference in hours between checkin & checkout vs count of instances for bookings with mobile for 10m bookings')
plt.show()
train_bookings_mobile.shape#.ix[(train_bookings_mobile['stay_span'] <= -1), 'stay_span'] = 0
train_bookings_mobile.ix[(train_bookings_mobile['stay_span'] <= -1), 'stay_span'] = 0
sns.countplot(y='stay_span', data=train_bookings_mobile)
sns.plt.title('stay_span wise destination distribution of 10m bookings')
plt.show()
sns.set(style="darkgrid")
ax = sns.countplot(x="stay_span", hue="is_mobile", data=train_bookings)
sns.set(style="darkgrid")
ax = sns.countplot(x="stay_span", hue="is_package", data=hotel_cluster_count)
