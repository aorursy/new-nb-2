#import the Libraries

import os
import pandas as pd
import numpy as np
import scipy as sp
import random
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
os.listdir('../input/trends-assessment-prediction/')
features = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv')
submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv')
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv')
reveal = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')
numbers = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')
fmri_mask = '../input/trends-assessment-prediction/fMRI_mask.nii'
# Installing the nilearn
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn import surface
smri = 'ch2better.nii'
mask_img = nl.image.load_img(fmri_mask)

def load_subject(filename, mask_img):
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)

    return subject_img
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    nlplt.plot_prob_atlas(subject_img, bg_img=smri, view_type='filled_contours',
                          draw_cross=False, title='All %d spatial maps' % num_components, threshold='auto')
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_stat_map(first_rsn)
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    for img in image.iter_img(rsn):
        # img is now an in-memory 3D img
        plotting.plot_stat_map(img, threshold=3)
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)     
    plotting.plot_glass_brain(first_rsn,display_mode='lyrz')
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_epi(first_rsn)
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_anat(first_rsn)
    print("-"*50)
files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_roi(first_rsn)
    print("-"*50)
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]
view = plotting.view_img_on_surf(stat_img, threshold='90%')
view.open_in_browser()
view
features.head()
features.info()
from sklearn.impute import KNNImputer
impute = KNNImputer(n_neighbors=40)
features = impute.fit_transform(features)
features = pd.DataFrame(features,columns = ['Id','age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'])
features
loading.head()
loading.info()
fnc.head()
fnc.info()
reveal.head()
reveal.info()
numbers.head()
numbers.info()
sns.heatmap(features.corr(),annot=True,linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
sns.heatmap(loading.corr(),annot=True,linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
#main or, target element in problem
target_col = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
sns.distplot(features['age'], ax=ax[0],rug=True, rug_kws={"color": "coral"},
                  kde_kws={"color": "royalblue", "lw": 1.5},
                  hist_kws={"histtype": "bar", "linewidth": 3,
                            "alpha": 1, "color": "coral"}).set_title('Age')

sns.distplot(features['domain1_var1'], ax=ax[1],rug=True, rug_kws={"color": "coral"},
                  kde_kws={"color": "royalblue", "lw": 1.5},
                  hist_kws={"histtype": "bar", "linewidth": 3,
                            "alpha": 1, "color": "coral"}).set_title('domain1_var1')

sns.distplot(features['domain1_var2'], ax=ax[2],rug=True, rug_kws={"color": "coral"},
                  kde_kws={"color": "royalblue", "lw": 1.5},
                  hist_kws={"histtype": "bar", "linewidth": 3,
                            "alpha": 1, "color": "coral"}).set_title('domain1_var2')

sns.distplot(features['domain2_var1'], ax=ax[3],rug=True, rug_kws={"color": "coral"},
                  kde_kws={"color": "royalblue", "lw": 1.5},
                  hist_kws={"histtype": "bar", "linewidth": 3,
                            "alpha": 1, "color": "coral"}).set_title('domain2_var1')

sns.distplot(features['domain2_var2'], ax=ax[4],rug=True, rug_kws={"color": "coral"},
                  kde_kws={"color": "royalblue", "lw": 1.5},
                  hist_kws={"histtype": "bar", "linewidth": 3,
                            "alpha": 1, "color": "coral"}).set_title('domain2_var2')

fig.suptitle('Target Visualization', fontsize=10)
train = features.merge(loading, on='Id', how='left')
train = train.merge(fnc, on='Id', how='left')
train.head()
X = train.drop(target_col, axis=1)
X.head()
X
y = features
y.head()
y
y.info()
submission['ID_num'] = submission['Id'].apply(lambda x: int(x.split('_')[0]))
test = pd.DataFrame({'Id': submission['ID_num'].unique()})
del submission['ID_num'];
test.head()
test = test.merge(loading, on='Id', how='left')
test = test.merge(fnc, on='Id', how='left')
test.head()
test
test.info()
import autokeras as ak
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold,cross_val_score
from tensorflow.keras.layers import Dropout
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

regressor  = ak.StructuredDataRegressor(max_trials=100,loss='mean_absolute_error',metrics='accuracy', seed=42,output_dim=5)
regressor.fit(x=X.iloc[:,1:], y=y.iloc[:,1:],validation_split=0.25,
                               callbacks=[History(),
                               EarlyStopping(monitor='val_loss', 
                               patience=30, 
                               restore_best_weights=True, 
                               verbose=1)])

from tensorflow.keras.models import load_model,save_model
model = regressor.export_model()
print(type(model))
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, expand_nested=True)
import tensorflow as tf
tf.keras.models.save_model(model,'C:/OpenClassRooms/projet 8/',include_optimizer=True,save_format='tf',)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold,cross_val_score
from tensorflow.keras.layers import Dropout
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Dense(1404,input_dim=1404,kernel_initializer='normal',activation='relu'))
model.add(Dense(702,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(702,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(702,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,kernel_initializer='normal'))

model.compile(loss='mean_absolute_error',optimizer='adam', metrics = ['accuracy'])

model.summary()
loaded_model = load_model('C:/OpenClassRooms/projet 8', custom_objects=ak.CUSTOM_OBJECTS)
history=model.fit(X.iloc[:,1:],y.iloc[:,1:],epochs=1000,batch_size=32,validation_split=0.25,verbose=2,
                  callbacks=[History(),
                             EarlyStopping(monitor='val_loss', 
                             patience=30, 
                             restore_best_weights=True, 
                             verbose=1)])
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
prediction = regressor.predict(test.iloc[:,1:])
prediction = pd.DataFrame(prediction)
prediction.columns = y.iloc[:,1:].columns
prediction.head(10)
pred = pd.DataFrame()

for target in target_col:
    value = pd.DataFrame()
    value['Id'] = [f'{c}_{target}' for c in test['Id'].values]
    value['Predicted'] = prediction[target]
    pred = pd.concat([pred, value])

pred.head()
submission
submission = pd.merge(submission, pred, on = 'Id')
submission = submission[['Id', 'Predicted_y']]
submission.columns = ['Id', 'Predicted']
submission.to_csv('submission_deeplearning_knnimputer.csv', index=False)
submission.head()

submission['Predicted'].hist()