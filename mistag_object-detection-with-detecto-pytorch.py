import torch

import os, sys, ast

import pandas as pd

import numpy as np

import glob

import shutil

import matplotlib.pyplot as plt

DATA = '/kaggle/input/global-wheat-detection/'

df=pd.read_csv(DATA+"train.csv")

df.bbox = df.bbox.apply(ast.literal_eval)

for i in range(len(df)):

    df.bbox.iloc[i][2]=df.bbox.iloc[i][0]+df.bbox.iloc[i][2] # xmax

    df.bbox.iloc[i][3]=df.bbox.iloc[i][1]+df.bbox.iloc[i][3] # ymax

df.sample(5)




files = glob.glob(DATA+"train/*")

train, validate, rest = np.split(files, [int(len(files)*0.2), int(len(files)*0.25)])

for i in range(len(train)):

    shutil.copy2(train[i], './train')

for i in range(len(validate)):

    shutil.copy2(validate[i], './validate')

len(train), len(validate), len(rest)






from detecto import utils, visualize, core



plt.rcParams['figure.figsize'] = (12.0, 12.0)



files = glob.glob("./train/*")

for i in range(len(files)):

    fid = files[i].replace('./train/', '').split('.')[0]

    bx = df[df.image_id == fid]

    if len(bx) > 0:

        boxes=torch.FloatTensor(bx.bbox.tolist())

        image = utils.read_image('./train/'+fid+'.jpg')

        visualize.show_labeled_image(image, boxes)

        break

# install Pascal VOC writer from dataset



from pascal_voc_writer import Writer



LABEL = "Wheat"

def create_voc(folder):

    files = glob.glob(folder+"/*")

    for i in range(len(files)):

        fid = files[i].replace(folder+'/','').split('.')[0]

        ldf=df[df.image_id == fid].reset_index()

        if len(ldf)> 0:

            width, height = ldf.width.iloc[0], ldf.height.iloc[0]

            writer = Writer(fid+'.jpg', width, height)

            for j in range(len(ldf)):

                writer.addObject(LABEL, 

                                 int(ldf.bbox.iloc[j][0]), 

                                 int(ldf.bbox.iloc[j][1]), 

                                 int(ldf.bbox.iloc[j][2]),

                                 int(ldf.bbox.iloc[j][3]))

            writer.save(folder+'/'+fid+'.xml')

        

create_voc("./validate")

create_voc("./train")
dataset = core.Dataset('./train/')

loader = core.DataLoader(dataset, batch_size=16, shuffle=True)

val_dataset = core.Dataset('./validate/')

model = core.Model([LABEL])

losses = model.fit(loader, val_dataset, epochs=2, learning_rate=0.001, lr_step_size=5, verbose=True)

print(losses)
#cleanup


tfiles = glob.glob(DATA+"test/*")

predf=pd.DataFrame(columns=['image_id', 'PredictionString'])

for i in range(len(tfiles)):

    image = utils.read_image(tfiles[i])

    fid = tfiles[i].replace(DATA+'test/','').split('.')[0]

    predictions = model.predict(image)

    labels, boxes, scores = predictions

    b=boxes.numpy().astype(int)

    s=scores.numpy()

    pstr=''

    for i in range(len(b)):

        p=[b[i][0], b[i][1], b[i][2]-b[i][0], b[i][3]-b[i][1]]

        pstr=pstr+str(s[i])+' '+str(p[0])+' '+str(p[1])+' '+str(p[2])+' '+str(p[3])+' '

    predf=predf.append({'image_id': fid, 'PredictionString': pstr}, ignore_index=True)
predf.to_csv('submission.csv', index=False)