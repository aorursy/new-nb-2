# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from tqdm import tqdm

import cv2

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

path = '../input/understanding_cloud_organization'

import keras

import keras.backend as K

from sklearn.model_selection import train_test_split
Train_Dir = '/kaggle/input/understanding_cloud_organization/train_images'

Test_Dir = 'kaggle/input/understanding_cloud_organization/test_images'

for img in tqdm(os.listdir(Train_Dir)):

    imgr = cv2.imread(os.path.join(Train_Dir,img))

    # print(imgr.shape)
train_csv_folder = '/kaggle/input/understanding_cloud_organization/train.csv'

train_csv = pd.read_csv(train_csv_folder)

train_csv.shape
train_csv['ImageId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[0])

train_csv['ClassId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[1])

train_csv['hasmask'] = ~train_csv['EncodedPixels'].isna()

mask_count_df = train_csv.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasmask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head(10)
train_csv.head(10)
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    #print(s)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')
def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles
fig = plt.figure(figsize=(25, 16))

for j, im_id in enumerate(np.random.choice(train_csv['ImageId'].unique(), 4)):

    for i, (idx, row) in enumerate(train_csv.loc[train_csv['ImageId'] == im_id].iterrows()):

        ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])

        im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")

        plt.imshow(im)

        mask_rle = row['EncodedPixels']

        #print(mask_rle)

        try: # label might not be there!

            mask = rle_decode(mask_rle)

            #print(mask.shape)

        except:

            mask = np.zeros((1400, 2100))

        plt.imshow(mask, alpha=0.6, cmap='Blues')

        ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['ClassId']}")
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='/kaggle/input/understanding_cloud_organization/train_images',

                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,

                 augment=False, n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.augment = augment

        self.n_classes = n_classes

        self.shuffle = shuffle

        

        self.random_state = random_state

        

        self.on_epoch_end()

        np.random.seed(self.random_state)



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            

            if self.augment:

                X, y = self.__augment_batch(X, y)

            print(X,y)

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        if self.reshape is None:

            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        else:

            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        if self.reshape is None:

            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        else:

            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img

    

    def __random_transform(self, img, masks):

        composition = albu.Compose([

            albu.HorizontalFlip(),

            albu.VerticalFlip(),

            albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15)

        ])

        

        composed = composition(image=img, mask=masks)

        aug_img = composed['image']

        aug_masks = composed['mask']

        

        return aug_img, aug_masks

    

    def __augment_batch(self, img_batch, masks_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ], masks_batch[i, ] = self.__random_transform(

                img_batch[i, ], masks_batch[i, ])

        

        return img_batch, masks_batch



BATCH_SIZE = 4



train_idx, val_idx = train_test_split(

    mask_count_df.index, random_state=2019, test_size=0.2

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count_df,

    target_df=train_csv,

    batch_size=BATCH_SIZE,

    reshape=(320, 480),

    augment=True,

    n_channels=3,

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    df=mask_count_df,

    target_df=train_csv,

    batch_size=BATCH_SIZE, 

    reshape=(320, 480),

    augment=False,

    n_channels=3,

    n_classes=4

)
from keras.layers import Dense

from keras.models import Model



from keras.optimizers import Adam, Nadam

from keras.callbacks import Callback, ModelCheckpoint

from keras.losses import binary_crossentropy

import albumentations as albu

import segmentation_models as sm
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
model = sm.Unet(

    'resnet18', 

    classes=4,

    input_shape=(320, 480, 3),

    activation='sigmoid'

)

model.compile(optimizer=Nadam(lr=0.0002), loss=bce_dice_loss, metrics=[dice_coef])

model.summary()
sub_df = pd.read_csv('/kaggle/input/understanding_cloud_organization/sample_submission.csv')

sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
os.listdir('/kaggle/input/resnet18')


resnetpath = '/kaggle/input/resnet18/modelefficient.h5'

model.load_weights(resnetpath)

test_df = []



for i in range(0, test_imgs.shape[0], 500):

    batch_idx = list(

        range(i, min(test_imgs.shape[0], i + 500))

    )



    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        dim=(350, 525),

        reshape=(320, 480),

        n_channels=3,

        base_path='/kaggle/input/understanding_cloud_organization/test_images',

        target_df=sub_df,

        batch_size=1,

        n_classes=4

    )



    batch_pred_masks = model.predict_generator(

        test_generator, 

        workers=1,

        verbose=1

    )



    for j, b in enumerate(batch_idx):

        filename = test_imgs['ImageId'].iloc[b]

        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks, reshape=(350, 525))



        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)
os.listdir('/kaggle/input/csvfiledl/SubmissionDLearningFinal.csv')
df = pd.read_csv('/kaggle/input/csvfiledl/SubmissionDLearningFinal.csv')
df.to_csv('FinaldeepLearning.csv',index = False)