import numpy as np
import pandas as pd
import csv
import scipy.io
import os
import matplotlib.pyplot as plt


mode = 'test'

img_dir = os.path.join('./img', mode)
pos_dir = os.path.join('./img', mode, '1')
neg_dir = os.path.join('./img', mode, '0')

if not os.path.exists(pos_dir):
    os.makedirs(pos_dir)
if not os.path.exists(neg_dir):
    os.makedirs(neg_dir)

data_root = "./data/test"
image_list = os.listdir(data_root)
length = len(image_list)
labels = []
record_length = []
for img_name in image_list:
    data_path = os.path.join(data_root, img_name)
    data = scipy.io.loadmat(data_path)
    sample = data['value'][0]
    label = data['label'][0][0]
    labels.append(label)
    data_length = len(sample)
    record_length.append(data_length)

max_length = np.max(record_length)

for img_name in image_list:
    data_path = os.path.join(data_root, img_name)
    data = scipy.io.loadmat(data_path)
    sample = data['value'][0]
    label = data['label'][0][0]
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot()
    ax.plot(sample)
    ax.axis('off')
    ax.set_xlim([0, max_length])
    fig.savefig(os.path.join(img_dir, str(label), os.path.basename(img_name).rsplit(".")[0] + ".png"))
    plt.close()

