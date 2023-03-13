import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()
lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}

def prep_data(market_data):
    # add asset code representation as int (as in previous kernels)
    market_data['assetCodeT'] = market_data['assetCode'].map(lbl)
    market_col = ['assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 
                        'returnsOpenPrevMktres10']
    # select relevant columns, fillna with zeros (where dropped in previous kernels that I saw)
    # getting rid of time, assetCode (keep int representation assetCodeT), assetName, universe
    X = market_data[market_col].fillna(0).values
    if "returnsOpenNextMktres10" in list(market_data.columns):#if training data
        up = (market_data.returnsOpenNextMktres10 >= 0).values
        r = market_data.returnsOpenNextMktres10.values
        universe = market_data.universe
        day = market_data.time.dt.date
        assert X.shape[0] == up.shape[0] == r.shape[0] == universe.shape[0] == day.shape[0]
    else:#observation data without labels
        up = []
        r = []
        universe = []
        day = []
    return X, up, r, universe, day
X, up, r, universe, day = prep_data(market_train)
# r, u and d are used to calculate the scoring metric on test
X_train, X_test, up_train, up_test, _, r_test, _, u_test, _, d_test = \
train_test_split(X, up, r, universe, day, test_size=0.25, random_state=99)
#from other kernel
from xgboost import XGBClassifier
xgb_market = XGBClassifier(n_jobs=4, n_estimators=200, max_depth=8, eta=0.1)
#xgb_market = XGBClassifier(n_jobs=4, n_estimators=100)
t = time.time()
print('Fitting Up')
xgb_market.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}s')
# distribution of confidence that will be used as submission
confidence_test = xgb_market.predict_proba(X_test)[:,1]*2 -1
print(accuracy_score(confidence_test>0,up_test))
plt.hist(confidence_test, bins='auto')
plt.title("predicted confidence")
plt.show()
# calculation of actual metric that is used to calculate final score
r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_test = mean / std
print(score_test)
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()
    # discard assets that are not scored
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_market_obs = prep_data(market_obs_df)[0]
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = xgb_market.predict_proba(X_market_obs)[:,1]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()
market_col = ['assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 
                        'returnsOpenPrevMktres10']
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(len(xgb_market.feature_importances_)), xgb_market.feature_importances_)
plt.xticks(range(len(xgb_market.feature_importances_)), market_col, rotation='vertical');