import pandas as pd

import numpy as np

import os

import cv2



# visualization

import matplotlib.pyplot as plt



# plotly offline imports

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly import subplots

import plotly.express as px

import plotly.figure_factory as ff

from plotly.graph_objs import *

from plotly.graph_objs.layout import Margin, YAxis, XAxis

init_notebook_mode()



# frequent pattern mining

from mlxtend.frequent_patterns import fpgrowth



# path where all the training images are

img_path = '../input/train_images/'
pd.read_csv('../input/train.csv').head()
# Replace NaN or no mask with -1

train_df = pd.read_csv('../input/train.csv').fillna(-1)



train_df.head()
# separate IDs into 2 columns 



train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) # index 0 after jpg_

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1]) # index 1 after jpg_
train_df.head()
# Group class & pixel 

train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], 

                                                                row['EncodedPixels']), axis = 1)



# Dict for all defect labels 

grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)
grouped_EncodedPixels
train_df.head()
print('Total number of unique images: %s' % len(train_df['ImageId'].unique()))



print('Unique images with at least one defect: %s' %len(train_df[train_df['EncodedPixels'] != -1]['ImageId'].unique()))



print('Total instances of defects or total defects for all images: %s' % len(train_df[train_df['EncodedPixels'] != -1]))
total_images = len(train_df['ImageId'].unique())

defective = len(train_df[train_df['EncodedPixels'] != -1]['ImageId'].unique())

print('defective proportion of images: %s' % (defective / total_images))
# 10 images with 2 defect classes 

xsamples = []



for i in grouped_EncodedPixels.iteritems():

    if(len([i[1] for x in i[1] if x[1] != -1]) == 2) and (len(xsamples) < 10):

        xsamples.append(i[0])
# Decode mask

    # rle-to-mask-converter 

def rle_to_mask(rle_string, height, width):

    '''

    convert RLE(run length encoding) string to numpy array

    

    Parameters:

    rle_string (str): string of rle encoded mask

    height (int): mask height

    width (int): mask width

    

    Return:

    numpy.array: numpy array of the mask

    '''

    

    rows, cols = height, width

    

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]

        rle_pairs = np.array(rle_numbers).reshape(-1,2)

        img = np.zeros(rows*cols, dtype=np.uint8)

        for index, length in rle_pairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img
# visualize steel image with 4 defect classes in separate cols

def viz_two_class_from_path(img_path, img_id, encoded_masks):

    '''

    visualize an image with two types of defects by plotting them on two columns

    with the defect overlayed on top of the original image.



    Parameters: 

    img_path (str): path of images

    img_id (str): image id or filename of the path

    encoded_masks (list): a list of strings of encoded masks 

    

    Returns: 

    matplotlib image plot in columns for two classes iwth defect

    '''

    

    img = cv2.imread(os.path.join(img_path, img_id))

    fig, ax = plt.subplots(nrows = 1, ncols = 2, 

                           sharey = True, figsize = (20,10))

    cmaps = ["Reds", "Blues", "Greens", "Purples"]

    axid = 0

    for idx, encoded_mask in enumerate(encoded_masks):

        class_id = idx + 1

        if encoded_mask == -1:

            pass

        else:

            mask_decoded = rle_to_mask(encoded_mask, 256, 1600)

            ax[axid].get_xaxis().set_ticks([])

            ax[axid].get_yaxis().set_ticks([])

            ax[axid].text(0.25, 0.25, 'Image Id: %s - Class Id: %s' % (img_id, class_id), fontsize=12)

            ax[axid].imshow(img)

            ax[axid].imshow(mask_decoded, alpha = 0.15, cmap = cmaps[idx])

            axid += 1
# visualize images grabbed with mask

for xsample in xsamples:

    img_id = xsamples

    mask_1, mask_2, mask_3, mask_4 = grouped_EncodedPixels[xsample]

    masks = [mask_1[1], mask_2[1], mask_3[1], mask_4[1]]

    viz_two_class_from_path(img_path, xsample, masks)
# visualize image with 4 defect classes in seperate columns

def viz_one_class_from_path(img_path, img_id, mask, class_id, text=None):

    '''

    visualize an image with two types of defects by plotting them on two columns

    with the defect overlayed on top of the original image.



    Parameters: 

    img_path (str): path of images

    img_id (str): image id or filename of the path

    encoded_mask (str): RLE mask

    class_id (str): class id of the defect

    

    Returns: 

    matplotlib image plot in columns for two classes with defect

    '''

    img = cv2.imread(os.path.join(img_path, img_id))

    mask_decoded = rle_to_mask(mask, 256, 1600)

    fig, ax = plt.subplots(figsize = (20,10))

    cmaps = ["Reds", "Blues", "Greens", "Purples"] # colors mapped in numerical order to classes in array  

    ax.get_xaxis().set_ticks([])

    ax.get_yaxis().set_ticks([])

    if text: 

        ax.text(0.25, 0.25, text, fontsize=12)

    ax.imshow(img)

    ax.imshow(mask_decoded, alpha = 0.15, cmap = cmaps[int(class_id)-1])



def viz_per_class(train_df, class_id, sample_size = 5):

    class_samples = train_df[(train_df['ClassId'] == class_id) & (train_df['EncodedPixels']!=-1)].sample(sample_size)

    class_img_ids = class_samples['ImageId'].values

    class_encoded_masks = class_samples['EncodedPixels'].values

    

    for img_id, mask in zip(class_img_ids, class_encoded_masks):

        viz_one_class_from_path(img_path, img_id, mask, class_id)
# Class 1 

viz_per_class(train_df, '1', 1) # class 1, 1 image
# Class 2

viz_per_class(train_df, '2', 1) # class 2, 1 image 
viz_per_class(train_df, '3', 1)
viz_per_class(train_df, '4', 1)
# Sum of pixels for the mask for each class id



train_df['mask_pixel_sum'] = train_df.apply(lambda x: rle_to_mask(x['EncodedPixels'],

                                                                  width = 1500,

                                                                  height = 300).sum(),

                                           axis = 1)
# Num of images with no label (defect masks)



# -1 = defective 

annotation_count = grouped_EncodedPixels.apply(lambda x: 1 if len([1 for y in x if y[1] != -1]) > 0 else 0).value_counts()



annotation_count_labels = ['No Label' if x == 0 else 'Label' for x in annotation_count.index]
# Num of defects per image 



defect_count_df = grouped_EncodedPixels.apply(lambda x: len([1 for y in x if y[1] != -1]))



defect_count_per_image = defect_count_df.value_counts()



defect_count_labels = defect_count_per_image.index
# Viz 



trace0 = Bar(x = annotation_count_labels, 

             y = annotation_count, 

             name = 'Labelled v. Not Labelled')



trace1 = Bar(x = defect_count_labels, y = defect_count_per_image, 

             name = 'Defects per image')



fig = subplots.make_subplots(rows = 1, cols = 2)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(height = 500, width = 1000, 

                     title = 'Defect labels & defect frequency per image')



iplot(fig)
rand_10_samples = defect_count_df[defect_count_df == 0].sample(10).index



for samp in rand_10_samples:

    viz_one_class_from_path(img_path, samp, -1, 0)
class_ids = ['1', '2', '3', '4']



mask_count_per_class = [train_df[(train_df['ClassId'] == class_id) &

                                 (train_df['mask_pixel_sum']!= 0)]['mask_pixel_sum'].count() for class_id in class_ids]



pixel_sum_per_class = [train_df[(train_df['ClassId'] == class_id) &

                                (train_df['mask_pixel_sum']!= 0)]['mask_pixel_sum'].sum() for class_id in class_ids]
fig = subplots.make_subplots(rows = 1, cols = 2,

                             specs = [[{'type':'domain'}, 

                                      {'type':'domain'}]])



fig.add_trace(Pie(labels = class_ids, 

                  values = mask_count_per_class, 

                  name = 'Freq'), 1, 1)



fig.add_trace(Pie(labels = class_ids,

                  values = pixel_sum_per_class,

                  name = 'Area'), 1, 2)



fig.update_traces(hole = .4, hoverinfo = 'label + percent + name')



fig.update_layout(title_text = 'Steel Defect Mask & Pixel Count',

                  annotations = [dict(text = 'Freq', 

                                      x = 0.18, y = 0.5, 

                                      font_size = 20, 

                                      showarrow = False), 

                                 dict(text = 'Area', 

                                      x = 0.8, y = 0.5,

                                      font_size = 20,

                                      showarrow = False)])



fig.show()
# Hist & boxplot of mask pixels per class Id



fig = px.histogram(train_df[train_df['mask_pixel_sum']!=0][['ClassId', 'mask_pixel_sum']],

                   x = 'mask_pixel_sum', y = 'ClassId', 

                   color = 'ClassId', marginal = 'box')



fig['layout'].update(title = 'Histogram & boxplot of mask pixels per class')



fig.show()
# Build function to convert encoded mask to mask 

    # AND count number of segments per defect

    

def count_segment(mask):

    '''

    Given a mask, count number of regions 

        

    Parameters:

    mask (numpy.array): numpy array of the mask

    

    Returns:

    int: number of segments

    '''

    

    if mask.sum() == 0:

        return 0 

    else:

        # opencv & threshold mechanism to calc contours

        _, threshold = cv2.threshold(mask, 240, 255, 

                                     cv2.THRESH_BINARY)

        _, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        

        # segment count

        for c in contours:

            segments_count = len(c)

            

        return segments_count
# count_segment to convert encoded mask to mask & count segments per defect

train_df['segments'] = train_df.apply(lambda r: count_segment(rle_to_mask(r['EncodedPixels'],

                                                                           height = 256,

                                                                           width = 1600)),

                                      axis = 1)



train_df['avg_mask_per_seg'] = (train_df['mask_pixel_sum'] / train_df['segments']).fillna(0)
# Check if function works 



for segments in range(0,6):

    samp = train_df[train_df['segments'] == segments].sample(1)

    encoded_pixels = samp['EncodedPixels'].values[0]

    image_id = samp['ImageId'].values[0]

    class_id = samp['ClassId'].values[0]

    segments = samp['segments'].values[0]

    viz_one_class_from_path(img_path, image_id, 

                            encoded_pixels, class_id, 

                            text='Number of Segments: %s' % segments)
# Scatter plot 



fig = px.scatter(train_df[train_df['mask_pixel_sum']!=0], 

                 x = 'mask_pixel_sum', y = 'segments', 

                 color = 'ClassId',

                 size = 'avg_mask_per_seg', 

                 hover_data = ['avg_mask_per_seg'])



fig = px.scatter(train_df[train_df['mask_pixel_sum']!=0], 

                 x = 'mask_pixel_sum',

                 y = 'segments', 

                 color = 'ClassId',

                 marginal_y = 'rug', marginal_x = 'histogram')



fig.show()
segment_labels_per_class = []

segment_values_per_class = []



for class_id in ['1','2','3','4']:

    segments_value_count = train_df[(train_df['mask_pixel_sum']!=0) & (train_df['ClassId'] == class_id)]['segments'].value_counts()

    segments_value_count = segments_value_count.reset_index()

    segments_value_count.columns = ['segments','segments_count']

    segments_5_10_count = segments_value_count[(segments_value_count['segments'] >= 5) & 

                                               (segments_value_count['segments'] <= 10)]['segments_count'].sum()

    segments_10_plus_count = segments_value_count[segments_value_count['segments'] >10]['segments_count'].sum()

    segment_keys = list(segments_value_count[segments_value_count['segments'] < 5]['segments'].values)

    segment_keys.append('5 - 10')

    segment_keys.append('10 +')

    segments_values = list(segments_value_count[segments_value_count['segments'] < 5]['segments_count'].values)

    segments_values.append(segments_5_10_count)

    segments_values.append(segments_10_plus_count)

    segment_labels_per_class.append(segment_keys)

    segment_values_per_class.append(segments_values)
# Create subplots: use 'domain' type for Pie subplot

fig = subplots.make_subplots(rows = 1, cols = 4, 

                             specs = [[{'type':'domain'}, {'type':'domain'},

                                       {'type':'domain'}, {'type':'domain'}]])



fig.add_trace(Pie(labels=segment_labels_per_class[0], 

                  values = segment_values_per_class[0], name="Class Id 1"), 1, 1)



fig.add_trace(Pie(labels=segment_labels_per_class[1], 

                  values = segment_values_per_class[1], name="Class Id 2"), 1, 2)



fig.add_trace(Pie(labels=segment_labels_per_class[2], 

                  values = segment_values_per_class[2], name="Class Id 3"), 1, 3)



fig.add_trace(Pie(labels=segment_labels_per_class[3], 

                  values = segment_values_per_class[3], name="Class Id 4"), 1, 4)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Steel Defect Segments Count",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='1', x=0.1, y = 0.5, 

                      font_size = 20, showarrow = False),

                 dict(text='2', x = 0.37, y = 0.5, 

                      font_size = 20, showarrow = False),

                 dict(text='3', x = 0.63, y = 0.5,

                      font_size=20, showarrow=False),

                 dict(text='4', x = 0.91, y = 0.5, font_size=20, showarrow=False)])



fig.show()
# Create subplots: use 'domain' type for Pie subplot

fig = subplots.make_subplots(rows=1, cols=4, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(Pie(labels=segment_labels_per_class[0], values=segment_values_per_class[0], name="Class Id 1"), 1, 1)

fig.add_trace(Pie(labels=segment_labels_per_class[1], values=segment_values_per_class[1], name="Class Id 2"), 1, 2)

fig.add_trace(Pie(labels=segment_labels_per_class[2], values=segment_values_per_class[2], name="Class Id 3"), 1, 3)

fig.add_trace(Pie(labels=segment_labels_per_class[3], values=segment_values_per_class[3], name="Class Id 4"), 1, 4)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Steel Defect Segments Count",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='1', x=0.1, y=0.5, font_size=20, showarrow=False),

                 dict(text='2', x=0.37, y=0.5, font_size=20, showarrow=False),

                 dict(text='3', x=0.63, y=0.5, font_size=20, showarrow=False),

                 dict(text='4', x=0.91, y=0.5, font_size=20, showarrow=False)])

fig.show()
# Series with defect classes



class_per_image = grouped_EncodedPixels.apply(lambda encoded_list: [x[0] for x in encoded_list if x[1] !=-1])
# list of class count 



class_per_image_list = []

for r in class_per_image.iteritems():

    class_count = {'1':0, '2':0, '3':0, '4':0}

    for image_class in r[1]:

        class_count[image_class] = 1

    class_per_image_list.append(class_count)
# Frequency Pattern calc. for all images

class_per_image_df = pd.DataFrame(class_per_image_list)



class_fp_df = fpgrowth(class_per_image_df, 

                       use_colnames = True, 

                       min_support = 0.001)



class_fp_df = class_fp_df.sort_values(by = ['support'])
# Images with at least 1 mask 



class_per_defect_image_df = class_per_image_df[(class_per_image_df.T != 0).any()]



class_fp_defect_df = fpgrowth(class_per_defect_image_df, 

                              use_colnames = True,

                             min_support = 0.001)



class_fp_defect_df = class_fp_defect_df.sort_values(by = ['support'])
def bar_h(x, y, title, x_label, y_label):

    y_pos = np.arange(len(y))

    plt.barh(y_pos, x)

    plt.yticks(y_pos, y)

    plt.title(title)

    plt.ylabel(y_label)

    plt.xlabel(x_label)
plt.figure(figsize=(15,5))



# plot for FP for all images

plt.subplot(1,2,1)

combinations = [', '.join(x) for x in class_fp_df['itemsets'].values]

support = class_fp_df['support'].values

bar_h(support, combinations, 'Defect Classes Appearing Frequently - All Samples', 'Defect Classes', 'Support')



# plot for FP for images with at least one fault

plt.subplot(1,2,2)

combinations = [', '.join(x) for x in class_fp_defect_df['itemsets'].values]

support = class_fp_defect_df['support'].values

bar_h(support, combinations, 'Defect Classes Appearing Frequently - Defective Samples Only', 'Defect Classes', 'Support')