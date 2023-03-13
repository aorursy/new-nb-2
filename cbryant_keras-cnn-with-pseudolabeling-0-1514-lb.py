import pandas as pd 

import numpy as np 

import cv2 # Used to manipulated the images 

np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though



# Import Keras 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam



from sklearn.model_selection import StratifiedShuffleSplit, KFold

from scipy.ndimage.filters import uniform_filter

from scipy.ndimage.measurements import variance
df_train = pd.read_json('../input/train.json') # this is a dataframe
def get_scaled_imgs(df):

    imgs = []

    

    for i, row in df.iterrows():

        #make 75x75 image

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        

        # use a lee filter to help with speckling

        band_1 = lee_filter(band_1,4)

        band_2 = lee_filter(band_2,4)

        band_3 = lee_filter(band_3,4)

        

        # Rescale

        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())

        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())

        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())



        imgs.append(np.dstack((a, b, c)))



    return np.array(imgs)
def lee_filter(img, size):

    

    img_mean = uniform_filter(img, (size, size))

    img_sqr_mean = uniform_filter(img**2, (size, size))

    img_variance = img_sqr_mean - img_mean**2



    overall_variance = variance(img)



    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)

    img_output = img_mean + img_weights * (img - img_mean)



    return img_output
Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)

idx_tr = np.where(df_train.inc_angle>0)
Ytrain = Ytrain[idx_tr[0]]

Xtrain = Xtrain[idx_tr[0],...]
df_test = pd.read_json('../input/test.json')

df_test.inc_angle = df_test.inc_angle.replace('na',0)

Xtest = (get_scaled_imgs(df_test))
def get_augment(imgs):

    

    more_images = []

    vert_flip_imgs = []

    hori_flip_imgs = []

      

    for i in range(0,imgs.shape[0]):

        a=imgs[i,:,:,0]

        b=imgs[i,:,:,1]

        c=imgs[i,:,:,2]

        

        av=cv2.flip(a,1)

        ah=cv2.flip(a,0)

        bv=cv2.flip(b,1)

        bh=cv2.flip(b,0)

        cv=cv2.flip(c,1)

        ch=cv2.flip(c,0)

        

        vert_flip_imgs.append(np.dstack((av, bv, cv)))

        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

      

    v = np.array(vert_flip_imgs)

    h = np.array(hori_flip_imgs)

       

    more_images = np.concatenate((imgs,v,h))

    

    return more_images

def getModel():

    #Build keras model

    

    model=Sequential()

    

    # CNN 1

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Dropout(0.2))



    # CNN 2

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    # CNN 3

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    #CNN 4

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    # You must flatten the data for the dense layers

    model.add(Flatten())



    #Dense 1

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))



    #Dense 2

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.2))



    # Output 

    model.add(Dense(1, activation="sigmoid"))



    optimizer = Adam(lr=0.001, decay=0.0)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    

    return model
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2)



for train_index, cv_index in sss.split(Xtrain, Ytrain):



    X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]

    y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]

    Xtr_more = get_augment(X_train) 

    Xcv_more = get_augment(X_cv) 

    Ytr_more = np.concatenate((y_train,y_train,y_train))

    Ycv_more = np.concatenate((y_cv,y_cv,y_cv))

    model = getModel()

    model.summary()



    batch_size = 32

    

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
    model.load_weights(filepath = '.mdl_wts.hdf5')



    score = model.evaluate(Xcv_more, Ycv_more, verbose=2)

    print('CV loss:', score[0])

    print('CV accuracy:', score[1])



    pt = model.predict(Xcv_more)

    mse = (np.mean((pt-Ycv_more)**2))

    print('CV MSE: ', mse)

    

    predA_test = model.predict(Xtest) # Here, we make the predictions for use in pseudo-labelling
    idx_pred_1 = (np.where(predA_test[:,0]>0.95))

    idx_pred_0 = (np.where(predA_test[:,0]<0.05))

    

    Xtrain_pl = np.concatenate((Xtrain,Xtest[idx_pred_1[0],...],Xtest[idx_pred_0[0],...]))

    Ytrain_pl = np.concatenate((Ytrain,np.ones(idx_pred_1[0].shape[0]),np.zeros(idx_pred_0[0].shape[0])))

    

    pl_kf = KFold(n_splits=5, shuffle=True)



    for train_pl_index, cv_pl_index in pl_kf.split(Xtrain_pl, Ytrain_pl):

        Xtrain_pl, Xpl_cv = Xtrain_pl[train_pl_index], Xtrain_pl[cv_pl_index]

        Ytrain_pl, Ypl_cv = Ytrain_pl[train_pl_index], Ytrain_pl[cv_pl_index]

        break #you can remove this to add more folds - set to one for demo

       

    model = getModel()

    

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    mcp_save = ModelCheckpoint('.mdl_wtsPL.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, epsilon=1e-4, mode='min')

    history_pl = model.fit(Xtrain_pl, Ytrain_pl, batch_size=batch_size, epochs=30, verbose=0, callbacks=[earlyStopping, mcp_save,reduce_lr_loss], validation_data=(Xpl_cv,Ypl_cv))

    
    model.load_weights(filepath = '.mdl_wtsPL.hdf5')

    

    scorePLCV = model.evaluate(Xpl_cv, Ypl_cv, verbose=0)

    print('Train PL CV score:', scorePLCV[0])

    print('Train PL CV accuracy:', scorePLCV[1])

    

    score = model.evaluate(Xtrain_pl, Ytrain_pl, verbose=0)

    print('Train PL score:', score[0])

    print('Train PL accuracy:', score[1])

    

    

    score = model.evaluate(X_cv, y_cv, verbose=0)

    print('X_cv score:', score[0])

    print('X_cv accuracy:', score[1])

    

    score = model.evaluate(Xtrain, Ytrain, verbose=0)

    print('Train score:', score[0])

    print('Train accuracy:', score[1])

    predA_test = model.predict(Xtest)

    

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predA_test.reshape((predA_test.shape[0]))})

    print(submission.head(10))

    

    submission.to_csv(INPUT_PATH + '20180101_submission'+'.csv', index=False)