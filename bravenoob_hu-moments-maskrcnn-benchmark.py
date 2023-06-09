# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
import os

import sys

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab



import requests

from io import BytesIO

from PIL import Image

import numpy as np
# this makes our figures bigger

pylab.rcParams['figure.figsize'] = 20, 12

os.chdir('cocoapi/PythonAPI')


os.chdir('../..')




os.chdir('apex')


os.chdir('..')




os.chdir('maskrcnn-benchmark')


os.chdir('..')
sys.path.append('/kaggle/working/cocoapi/PythonAPI')

sys.path.append('/kaggle/working/apex')

sys.path.append('/kaggle/working/maskrcnn-benchmark')

sys.path.append('/kaggle/working/maskrcnn-benchmark/demo')

from maskrcnn_benchmark.config import cfg

from predictor import COCODemo
# set up demo for keypoints

config_file = "maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"

cfg.merge_from_file(config_file)

cfg.merge_from_list(["MODEL.DEVICE", "cuda"])



coco_demo = COCODemo(

    cfg,

    min_image_size=400,

    confidence_threshold=0.9,

)
def load(url):

    pil_image = Image.open(url).convert("RGB")

    # convert to BGR format

    image = np.array(pil_image)[:, :, [2, 1, 0]]

    return image



def imshow(img):

    plt.imshow(img[:, :, [2, 1, 0]])

    plt.axis("off")
def calculate_hu(df, image_path, show_image=False): 

    head, filename = os.path.split(image_path)

    image = load(image_path)



    if show_image:

        # Visualize results

        predictions = coco_demo.run_on_opencv_image(image)

        imshow(predictions)

        return

        

    predictions = coco_demo.compute_prediction(image)

    predictions = coco_demo.select_top_predictions(predictions)

    

    number_of_cats = int((predictions.get_field("labels").numpy() == 16).sum())

    number_of_dogs = int((predictions.get_field("labels").numpy() == 17).sum())



    cat_mask_index = np.where(predictions.get_field("labels").numpy() == 16)[0]

    dog_mask_index = np.where(predictions.get_field("labels").numpy() == 17)[0]

    

    new_row = pd.Series({"imageId": filename,"cats": number_of_cats, "dogs": number_of_dogs})

    

    moments = 0

    moments_limit = 10

       

    for index in cat_mask_index:

        if moments >= moments_limit:

            break

        M = cv.moments(predictions.get_field("mask").numpy()[index][0])

        huMoments = cv.HuMoments(M)

        i = str(moments)

        hulist = pd.Series({"hu1_" + i: huMoments[0][0],"hu2_" + i: huMoments[1][0], "hu3_" + i: huMoments[2][0], "hu4_" + i: huMoments[3][0],

                           "hu5_" + i: huMoments[4][0], "hu6_" + i: huMoments[5][0], "hu7_" + i: huMoments[6][0]})

        new_row = new_row.append(hulist)

        moments = moments + 1



    for index in dog_mask_index:

        if moments >= moments_limit:

            break

        M = cv.moments(predictions.get_field("mask").numpy()[index][0])

        huMoments = cv.HuMoments(M)

        i = str(moments)

        hulist = pd.Series({"hu1_" + i: huMoments[0][0],"hu2_" + i: huMoments[1][0], "hu3_" + i: huMoments[2][0], "hu4_" + i: huMoments[3][0],

                           "hu5_" + i: huMoments[4][0], "hu6_" + i: huMoments[5][0], "hu7_" + i: huMoments[6][0]})

        new_row = new_row.append(hulist)

        moments = moments + 1

    

    return df.append(new_row, ignore_index=True)
# Directory of images to run detection on

IMAGE_DIR = '../input/petfinder-adoption-prediction/train_images'
pet_hu_moments=pd.DataFrame()

failed_images = []

counter = 0

for filename in os.listdir(IMAGE_DIR):

    try:

        image_number = filename[filename.find('-')+1 :-4]

        if(image_number not in ['1','2','3', '4']):

            continue   

        counter = counter +1

        if counter % 1000 == 0:

            print(counter)

        pet_hu_moments = calculate_hu(pet_hu_moments,os.path.join(IMAGE_DIR, filename))

    except Exception as e:

        print(e)

        failed_images.append(filename)

        continue
for img in failed_images:

    print(img)
pet_hu_moments.shape
pet_hu_moments.describe()
pet_hu_moments.fillna(0, inplace=True)
#show images with most cats and dogs in them

max_dog_count_petid = pet_hu_moments.loc[pet_hu_moments['dogs'].idxmax()]['imageId']

max_cat_count_petid = pet_hu_moments.loc[pet_hu_moments['cats'].idxmax()]['imageId']



calculate_hu(pd.DataFrame(),os.path.join(IMAGE_DIR, max_cat_count_petid), show_image=True)
calculate_hu(pd.DataFrame(),os.path.join(IMAGE_DIR, max_dog_count_petid), show_image=True)
def add_image_segmentations(df, PetID):

    #df = area cover df with imageId

    matching_rows = df[df['imageId'].str.contains(PetID)]

    count_row = matching_rows.shape[0]

        

    if count_row == 0:

        # no images for this pet

        return pd.Series()

    

    matching_rows['imageIndex'] = matching_rows.apply(lambda x: x['imageId'][x['imageId'].find('-')+1 :-4], axis=1)

    matching_rows = matching_rows.sort_values('imageIndex')

        

    # add values from all images

    pet_photo_values = pd.Series()

    for i, row in matching_rows.iterrows():

        img_nr = row['imageIndex']

        pet_photo_values = pet_photo_values.append(pd.Series(

            [row['cats'], row['dogs'], 

             row['hu1_0'], row['hu2_0'],row['hu3_0'], row['hu4_0'], row['hu5_0'], row['hu6_0'], row['hu7_0'], 

             row['hu1_1'], row['hu2_1'],row['hu3_1'], row['hu4_1'], row['hu5_1'], row['hu6_1'], row['hu7_1'],

             row['hu1_2'], row['hu2_2'],row['hu3_2'], row['hu4_2'], row['hu5_2'], row['hu6_2'], row['hu7_2'],

             row['hu1_3'], row['hu2_3'],row['hu3_3'], row['hu4_3'], row['hu5_3'], row['hu6_3'], row['hu7_3'], 

             row['hu1_4'], row['hu2_4'],row['hu3_4'], row['hu4_4'], row['hu5_4'], row['hu6_4'], row['hu7_4'],

             row['hu1_5'], row['hu2_5'],row['hu3_5'], row['hu4_5'], row['hu5_5'], row['hu6_5'], row['hu7_5'],           

             row['hu1_6'], row['hu2_6'],row['hu3_6'], row['hu4_6'], row['hu5_6'], row['hu6_6'], row['hu7_6'], 

             row['hu1_7'], row['hu2_7'],row['hu3_7'], row['hu4_7'], row['hu5_7'], row['hu6_7'], row['hu7_7'],

             row['hu1_8'], row['hu2_8'],row['hu3_8'], row['hu4_8'], row['hu5_8'], row['hu6_8'], row['hu7_8'],

             row['hu1_9'], row['hu2_9'],row['hu3_9'], row['hu4_9'], row['hu5_9'], row['hu6_9'], row['hu7_9']            

            ],                       

             index = ['cats_'+img_nr, 'dogs_'+img_nr, 

                    'hu1_0'+img_nr, 'hu2_0'+img_nr, 'hu3_0'+img_nr, 'hu4_0'+img_nr,'hu5_0'+img_nr, 'hu6_0'+img_nr, 'hu7_0'+img_nr,

                    'hu1_1'+img_nr, 'hu2_1'+img_nr, 'hu3_1'+img_nr, 'hu4_1'+img_nr,'hu5_1'+img_nr, 'hu6_1'+img_nr, 'hu7_1'+img_nr,

                    'hu1_2'+img_nr, 'hu2_2'+img_nr, 'hu3_2'+img_nr, 'hu4_2'+img_nr,'hu5_2'+img_nr, 'hu6_2'+img_nr, 'hu7_2'+img_nr,

                    'hu1_3'+img_nr, 'hu2_3'+img_nr, 'hu3_3'+img_nr, 'hu4_3'+img_nr,'hu5_3'+img_nr, 'hu6_3'+img_nr, 'hu7_3'+img_nr,

                    'hu1_4'+img_nr, 'hu2_4'+img_nr, 'hu3_4'+img_nr, 'hu4_4'+img_nr,'hu5_4'+img_nr, 'hu6_4'+img_nr, 'hu7_4'+img_nr,

                    'hu1_5'+img_nr, 'hu2_5'+img_nr, 'hu3_5'+img_nr, 'hu4_5'+img_nr,'hu5_5'+img_nr, 'hu6_5'+img_nr, 'hu7_5'+img_nr,

                    'hu1_6'+img_nr, 'hu2_6'+img_nr, 'hu3_6'+img_nr, 'hu4_6'+img_nr,'hu5_6'+img_nr, 'hu6_6'+img_nr, 'hu7_6'+img_nr,

                    'hu1_7'+img_nr, 'hu2_7'+img_nr, 'hu3_7'+img_nr, 'hu4_7'+img_nr,'hu5_7'+img_nr, 'hu6_7'+img_nr, 'hu7_7'+img_nr,

                    'hu1_8'+img_nr, 'hu2_8'+img_nr, 'hu3_8'+img_nr, 'hu4_8'+img_nr,'hu5_8'+img_nr, 'hu6_8'+img_nr, 'hu7_8'+img_nr,

                    'hu1_9'+img_nr, 'hu2_9'+img_nr, 'hu3_9'+img_nr, 'hu4_9'+img_nr,'hu5_9'+img_nr, 'hu6_9'+img_nr, 'hu7_9'+img_nr

                     ] ))

        

    return pet_photo_values
train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
result_df=pd.DataFrame()

result_df['PetID']=train_df['PetID']

result_df['AdoptionSpeed']=train_df['AdoptionSpeed']

result_df = result_df.merge(result_df.PetID.apply(lambda x: add_image_segmentations(pet_hu_moments, x)), left_index=True, right_index=True)

result_df.fillna(0, inplace=True)
result_df.shape
result_df.columns
result_df.to_csv('train_hu_moments.csv', index=False)
from sklearn.decomposition import TruncatedSVD



# svd

svd = TruncatedSVD(n_components=8)

result = svd.fit_transform(result_df.drop(['PetID','AdoptionSpeed'], axis=1))



result = pd.DataFrame(data=result[0:,0:],

                  columns=['hu_svd1','hu_svd2', 'hu_svd3','hu_svd4','hu_svd5','hu_svd6','hu_svd7','hu_svd8' ])



result.shape
result.columns
result['PetID']=train_df['PetID']

result['AdoptionSpeed']=train_df['AdoptionSpeed']
result.to_csv('train_hu_moments_svd.csv', index=False)
explained_variance=svd.explained_variance_

explained_variance_ratio=svd.explained_variance_ratio_

print("\n\nExplained Variance ",explained_variance)

print("\n\nExplained Variance Ratio  ", explained_variance_ratio)

print("\n\nCummulative Sum  ", np.cumsum(svd.explained_variance_))

print("\n\nExplained Variance Ratio Sum ",svd.explained_variance_ratio_.sum())  

print("\n\nSingular Values ",svd.singular_values_)

 



plt.subplot(1, 2, 2)

plt.bar(range(8), explained_variance_ratio, alpha=0.5, align='center', label='Explained variance ratio')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.title("Explained Variance Ratio", fontsize=20)

plt.show()
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: 

    XGBClassifier(),

    

    #CatBoostClassifier(verbose=0)

    ]

def train_model(data, MLA_list = MLA):

    

    target = data['AdoptionSpeed']

    X_train = data.drop(['AdoptionSpeed'],axis=1)

    

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA cohen_kappa_score','MLA Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)



    MLA_predict = data['AdoptionSpeed']

    

    row_index = 0

    for alg in MLA_list:



        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

        kf = StratifiedKFold(n_splits=5, shuffle=True)

        kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')

        cv_results = model_selection.cross_validate(alg, X_train, target, cv  = kf, scoring=kappa_score )

        

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

        MLA_compare.loc[row_index, 'MLA cohen_kappa_score'] = cv_results['test_score'].mean() 

        MLA_compare.loc[row_index, 'algo'] = alg.__class__

             

        #MLA_predict[MLA_name] = alg.predict(X_train)

        row_index+=1



    MLA_compare.sort_values(by = ['MLA cohen_kappa_score'], ascending = False, inplace = True)

    sns.barplot(x='MLA cohen_kappa_score', y = 'MLA Name', data = MLA_compare, color = 'b')

    plt.title('Machine Learning Algorithm Accuracy Score \n')

    plt.xlabel('Accuracy Score (%)')

    plt.ylabel('Algorithm')

    

    return MLA_compare
classifier_performance = train_model(result.drop(['PetID'], axis=1))
X_test = result.drop(['PetID', 'AdoptionSpeed'], axis=1)

target = result['AdoptionSpeed']



best_classifier = classifier_performance.iloc[0][4]



classifier = best_classifier().fit(X_test, target)
# Directory of images to run detection on

TEST_IMAGE_DIR = '../input/petfinder-adoption-prediction/test_images'

test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
test_pet_area_coverage=pd.DataFrame()

counter = 0

for filename in os.listdir(TEST_IMAGE_DIR):

    try:

        image_number = filename[filename.find('-')+1 :-4]

        if(image_number not in ['1','2','3', '4']):

            continue   

        counter = counter +1

        if counter % 1000 == 0:

            print(counter)

        test_pet_area_coverage = calculate_hu(test_pet_area_coverage,os.path.join(TEST_IMAGE_DIR, filename))

    except Exception:

        continue
test_result_df=pd.DataFrame()

test_result_df['PetID']=test_df['PetID']

test_result_df = test_result_df.merge(test_result_df.PetID.apply(lambda x: add_image_segmentations(test_pet_area_coverage, x)), left_index=True, right_index=True)
col_list = result_df.columns.tolist()

test_result_df = test_result_df.loc[:, col_list].fillna(0)

# unify column order

test_result_df = test_result_df[result_df.columns.drop('AdoptionSpeed')]
test_result = svd.transform(test_result_df.drop(['PetID'], axis=1))



test_result = pd.DataFrame(data=test_result[0:,0:],

                  columns=['hu_svd1','hu_svd2', 'hu_svd3','hu_svd4','hu_svd5','hu_svd6','hu_svd7','hu_svd8' ])

test_result['PetID']=test_df['PetID']
test_result.to_csv('test_hu_moments.csv', index=False)

test_result.to_csv('test_hu_moments_svd.csv', index=False)
submit=pd.DataFrame()

submit['PetID']=test_result['PetID']

submit['AdoptionSpeed']=classifier.predict(test_result.drop(['PetID'], axis=1))

submit['AdoptionSpeed']=submit['AdoptionSpeed'].astype(int)

submit.to_csv('submission.csv',index=False)
X_train = result.drop(['PetID'], axis=1)

X_test = test_result.drop(['PetID'], axis=1)
print(X_train.shape)

print(X_test.shape)
xgb_params = {

    'eval_metric': 'rmse',

    'seed': 1337,

    'silent': 1,

}
import xgboost as xgb



def run_xgb(params, X_train, X_test):

    n_splits = 5

    verbose_eval = 1000

    num_rounds = 30000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = X_tr['AdoptionSpeed'].values

        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)



        y_val = X_val['AdoptionSpeed'].values

        X_val = X_val.drop(['AdoptionSpeed'], axis=1)



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test
model, oof_train, oof_test = run_xgb(xgb_params, X_train, X_test)
# from https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved



import scipy as sp



from collections import Counter

from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix





# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(oof_train, X_train['AdoptionSpeed'].values)

coefficients = optR.coefficients()

print(coefficients)

valid_pred = optR.predict(oof_train, coefficients)

qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)

print("QWK = ", qwk)
coefficients_ = coefficients.copy()

#coefficients_[0] = 1.65

train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)

test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
submission = pd.DataFrame({'PetID': test_result['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission_opt.csv', index=False)
import shutil

shutil.rmtree('cocoapi')

shutil.rmtree('apex')

shutil.rmtree('maskrcnn-benchmark')