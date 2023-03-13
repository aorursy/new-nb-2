
# Install facenet-pytorch




# Copy model checkpoints to torch cache so they are loaded automatically by the package



import os

import glob

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Load face detector

mtcnn = MTCNN(device=device).eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

filenames
# Get all test videos

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')



# Number of frames to sample (evenly spaced) from each video

n_frames = 10



X = []

with torch.no_grad():

    for i, filename in enumerate(filenames):

        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')

        

        try:

            # Create video reader and find length

            v_cap = cv2.VideoCapture(filename)

            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            

            # Pick 'n_frames' evenly spaced frames to sample

            sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)

            imgs = []

            for j in range(v_len):

                success, vframe = v_cap.read()

                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

                if j in sample:

                    imgs.append(Image.fromarray(vframe))

            v_cap.release()

            

            # Pass image batch to MTCNN as a list of PIL images

            faces = mtcnn(imgs)

            

            # Filter out frames without faces

            faces = [f for f in faces if f is not None]

            faces = torch.stack(faces).to(device)

            

            # Generate facial feature vectors using a pretrained model

            embeddings = resnet(faces)

            

            # Calculate centroid for video and distance of each face's feature vector from centroid

            centroid = embeddings.mean(dim=0)

            X.append((embeddings - centroid).norm(dim=1).cpu().numpy())

        except KeyboardInterrupt:

            raise Exception("Stopped.")

        except:

            X.append(None)
#bias = -0.2942

#weight = 0.068235746



submission = []

for filename, x_i in zip(filenames, X):

    prob = np.prod(x_i)

    #if x_i is not None and len(x_i) == 10:

        #prob = 1 / (1 + np.exp(-(bias + (weight * x_i).sum())))

    #else:

        #prob = 0.5

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)
plt.hist(submission.label, 20)

plt.show()

submission