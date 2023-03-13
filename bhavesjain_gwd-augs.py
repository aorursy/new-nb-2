import pandas as pd
import numpy as np
import cv2
import os
import torch
from torchvision import models,transforms

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
import time
from skimage.util import random_noise
from skimage.filters import gaussian
# %cd /kaggle/working/
# !git clone https://github.com/cocodataset/cocoapi.git
# %cd /kaggle/working/cocoapi/PythonAPI
# !python setup.py build_ext install

# %cd /kaggle/working/

# !pip install cython
# # Install pycocotools, the version by default in Colab
# # has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# # Download TorchVision repo to use some files from
# # references/detection
# !git clone https://github.com/pytorch/vision.git
# %cd vision
# !git checkout v0.3.0

# !cp references/detection/utils.py ../
# !cp references/detection/transforms.py ../
# !cp references/detection/coco_eval.py ../
# !cp references/detection/engine.py ../
# !cp references/detection/coco_utils.py ../
# from engine import train_one_epoch, evaluate
# import utils
# import transforms as T

from data_aug.data_aug import *
from data_aug.bbox_util import *
train_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')
print(train_df.shape)
img_ids = train_df['image_id'].unique()
valid_ids = img_ids[-100:]
train_ids = img_ids[:-100]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape,train_df.shape
# a = cv2.imread("/kaggle/input/global-wheat-detection/test/2fd875eaa.jpg")
# plt.figure(figsize=(20,10))
# plt.imshow(a)
# # plt.figure(figsize=(20,10))
# # plt.imshow(gaussian(a,sigma=15))
# plt.figure(figsize=(20,10))
# plt.imshow(random_noise(a, mode='pepper',amount=0.2))#'gaussian'))

# plt.figure(figsize=(20,10))
# plt.imshow(random_noise(a, mode='gaussian',mean = 0.2))
# b = gaussian(a,sigma = 15)
# c = random_noise(a, mode='gaussian')
class Sequence(object):
    def __init__(self, augmentations, probs = 0.25):
        self.augmentations = augmentations
        self.probs = probs
    def __call__(self, image, bboxes):
#         bboxes = target["boxes"]
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
            img = image.copy()
            boxes = bboxes.copy()
            if random.random() < prob:
                img, boxes = augmentation(img, boxes)
        return img, boxes
# t1 = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
    
t2 = Sequence([RandomHSV(30, 0, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
class Blur(object):
    def __init__(self):
        '''
        initialize your transformation here, if necessary
        '''
      

    def __call__(self, tensor):
        arr = np.asarray(tensor) # transform to np array
        # apply your transformation
        arr = gaussian(arr,sigma=5)
#         arr = random_noise(arr, mode='gaussian',mean = 0.2)
#         pic = Image.fromarray(arr)
        return np.uint8(arr)
class Norm(object):
    def __init__(self):
        '''
        initialize your transformation here, if necessary
        '''

    def __call__(self, tensor):
        arr = np.asarray(tensor) # transform to np array
        # apply your transformation
#         arr = arr/255
        return np.uint8(arr)
class GWDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, train=False):
        super().__init__()

        self.img_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.train = train

    def __getitem__(self, index):
        img_idx = self.img_ids[index]
        img_name = str(img_idx+'.jpg')
        # load images ad masks
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
       # get bounding box coordinates for each mask
        num_bbxs = len(self.df[self.df['image_id']==img_idx])
        bbxs = self.df[self.df['image_id']==img_idx]
        boxes = []
        area = []
#         print(bbxs)
        for t in range(num_bbxs):
            l = bbxs.iloc[t]['bbox'].split(',')
#             print(l)
            xmin,ymin,w,h = float(l[0][1:]),float(l[1][1:]),float(l[2][1:]),float(l[3][1:-1])
            xmax = xmin+w
            ymax = ymin+h
            area.append(w*h)
            boxes.append([xmin, ymin, xmax, ymax])

        # there is only one class
        labels = torch.ones((num_bbxs,), dtype=torch.int64)

        imag_id = torch.tensor([index])
        # suppose all instances are not crowd
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((num_bbxs,), dtype=torch.int64)
#         print(boxes)
        target = {}
#         target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = imag_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        imge = img.copy()
        if self.train is True:
            imge,boxes = t2(imge, np.array(boxes))
#         imge = transforms.Compose([transforms.ToPILImage() ,transforms.RandomApply([MySkimageTransform()],p=0.5),transforms.ColorJitter(hue=.1, saturation=.1),transforms.ToTensor()])(np.uint8(imge))
#         imge = transforms.Compose([transforms.RandomChoice([Blur(),Norm()]),transforms.ToTensor()])(imge)
        imge = transforms.Compose([transforms.RandomChoice([Blur(),transforms.Compose([Norm(),transforms.ToPILImage(),transforms.ColorJitter(hue=.1, saturation=.1)])]),transforms.ToTensor()])(imge)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target["boxes"] = boxes
        return imge, target

    def __len__(self):
        return (self.img_ids.shape[0])
dataset = GWDataset(valid_df,'/kaggle/input/global-wheat-detection/train/',train = True)
# dataset[0]

for i in range(4):
    a = dataset[i][0].permute(1,2,0)
    b = dataset[i+1][0].permute(1,2,0)
    fig,(ax1, ax2) = plt.subplots(1, 2,figsize = (10,20))
    ax1.imshow(a)
    ax2.imshow(b)
    i=i+1
# model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# model1.load_state_dict(torch.load("/kaggle/input/gwd-train/fasterrcnn_resnet50_fpn_statedict.pth",map_location=torch.device('cpu')))
# model.eval()
model = torch.load("/kaggle/input/gwd-train/fasterrcnn_resnet50_fpn_new.pth",map_location=torch.device('cpu'))
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
def collate_fn(batch):
    return tuple(zip(*batch))

# split the dataset in train and test set

DIR_TRAIN = '/kaggle/input/global-wheat-detection/train/'
train_dataset = GWDataset(train_df, DIR_TRAIN,True)
valid_dataset = GWDataset(valid_df, DIR_TRAIN, False)
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
model.train()
model.to(device)#,dtype=torch.float)
num_epochs = 2
train_loss_min = 0.61
loss_hist = Averager()
itr = 1
total_train_loss = []
for epoch in range(num_epochs):
    loss_hist.reset()
    train_loss = []
    start_time = time.time()
    for images, targets in train_data_loader:
#         try:
#             print("!")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss.append(loss_value)
            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
#         except:
#             print("F",itr)
#             pass
    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    print(f'Epoch train loss is {epoch_train_loss}')
    if epoch_train_loss <= train_loss_min:
            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))
            # save checkpoint as best model
            torch.save(model, '/kaggle/working/fasterrcnn_resnet50_fpn_new'+str(epoch)+'.pth')
            train_loss_min = epoch_train_loss
    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, valid_data_loader, device=device)
# pick one image from the test set
for i in range(1):
    img, _ = valid_dataset[i]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device,dtype = torch.float32)])
    sample = valid_dataset[i][0].permute(1,2,0).numpy()
    boxes = prediction[0]['boxes'].cpu().numpy().astype(np.int32)
    # boxe = boxes.reshape((4,-1))
    scores = prediction[0]['scores'].cpu().numpy()
    im = np.array(img.permute(1,2,0))
    # plt.imshow(sample)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    color = (220,0,0)
    for i in range(len(boxes)):
    #     print(boxes[i])
        if scores[i]>0.90:
            cv2.rectangle(im,(int(boxes[i][0]), int(boxes[i][1])),(int(boxes[i][2]), int(boxes[i][3])),color, 5)
    ax.set_axis_off()
    ax.imshow(im)
dir_test = "/kaggle/input/global-wheat-detection/test"
preprocess = transforms.Compose([transforms.ToTensor()])
color = (220,0,0)
results = []
for img_file in os.listdir(dir_test):
    result = []
    img = cv2.imread(os.path.join(dir_test,img_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img/255.0
    img_t = preprocess(img)
#     img_t = img_t.unsqueeze(0)
    pred = model([img_t.to(device,dtype = torch.float32)])
    bboxes = pred[0]['boxes'].cpu().detach().numpy()
    bscores = pred[0]['scores'].cpu().detach().numpy()
    img_name = img_file.split('.')[:-1]
    for i in range(len(bboxes)):
        if bscores[i]>0.4:
            result.append((bscores[i],bboxes[i]))
    results.append((str(img_name[0]),result))
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
#     for box in bboxes:
#         cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color, 5)
    for i in result:
        if i[0]>0.4:
            box = i[1]
            cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color, 5)
    plt.figure()
    ax.set_axis_off()
    ax.imshow(img)
    plt.show()
torch.save(model, '/kaggle/working/fasterrcnn_resnet50_fpn_gwd_augs.pth')
torch.save(model.state_dict(), '/kaggle/working/fasterrcnn_resnet50_fpn_statedict.pth')
res = []
for result in results:
#     print(result[0],end='')
    pred_str = []
    for box in result[1]:
        pred_str.append(box[0])
        pred_str.append(box[1][0])
        pred_str.append(box[1][1])
        pred_str.append(box[1][2]-box[1][0])
        pred_str.append(box[1][3]-box[1][1])
    pred = {}
    pred['image_id'] = str(result[0])
    pred['PredictionString'] = ' '.join(str(i) for i in pred_str)
    res.append(pred)
test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])
print(test_df)
test_df.to_csv("/kaggle/working/submission.csv",index=False)
