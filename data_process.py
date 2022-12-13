import numpy as np
import pandas as pd
import csv
import scipy.io
import os
from sklearn import datasets
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
def sample_preprocessing(data, mu, sigma):
    # TODO: Do some fancy pre-process
    # Write your code here
    """
    pre-process your data
    :param data:
    :return:
    """

    # plt.figure()
    # plt.plot(data[0:2400])
    # data = (data-np.min(data))/(np.max(data)-np.min(data))
    nyq_rate = 300 / 2.0
    a = signal.firwin(512,60.0/nyq_rate)
    data = signal.filtfilt(a,1,data)
    data = signal.resample_poly(data,120,300)
    # mu = np.mean(data,axis=0)
    # sigma = np.std(data,axis=0)
    # data = (data-mu)/sigma



    # plt.figure()
    # plt.plot(data[0:2400])
    # plt.show()

    return data

if __name__ == '__main__':
    # Prepare data for training
    # Read csv file to record file list
    csv_reader = csv.reader(
        open("/home/huangdehao/github_projects/Proj2-ml/data/train_data.csv"))
    # Set training data directory
    train_dir = '/home/huangdehao/github_projects/Proj2-ml/data/train/'
    train_without_preprocess_dir = '/home/huangdehao/github_projects/Proj2-ml/data_without_preprocess'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(train_without_preprocess_dir):
        os.makedirs(train_without_preprocess_dir)

    # Read csv file by rows
    for row in csv_reader:
        filename = row[0]
        label_name = row[1]

        # Convert label from str to int
        if label_name == 'N':
            label = 1
        elif label_name == 'A':
            label = 0
        else:
            print('Unrecognizable label')
            continue
        # Read original data
        record = '/home/huangdehao/github_projects/Proj2-ml/ori_data/training2017/' + filename
        # Read waveform samples (input is in WFDB-MAT format)
        mat_data = scipy.io.loadmat(record + ".mat")
        samples = mat_data['val']
        samples = samples[0, :]
        print(samples)
        # preprocessing
        # TODO: Use given function to preprocess data
        new_samples = sample_preprocessing(samples)
        print(new_samples)
        # Save data
        save_filename = os.path.join(train_dir, filename + '.mat')
        scipy.io.savemat(save_filename, {'value': new_samples, 'label': label})
        save_filename = os.path.join(train_without_preprocess_dir, filename + '.mat')
        scipy.io.savemat(save_filename, {'value': samples, 'label': label})
        print(save_filename)
        # Prepare data for testing
    csv_reader = csv.reader(open(
        "/home/huangdehao/github_projects/Proj2-ml/data/test_data.csv"))  # Read csv file to record file list
    test_dir = '/home/huangdehao/github_projects/Proj2-ml/data/test/'
    test_without_preprocess_dir = '/home/huangdehao/github_projects/Proj2-ml/data_without_preprocess'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(test_without_preprocess_dir):
        os.makedirs(test_without_preprocess_dir)
    for row in csv_reader:
        filename = row[0]
        label_name = row[1]
        if label_name == 'N':
            label = 1
        elif label_name == 'A':
            label = 0
        else:
            print('Unrecognizable label ', label_name)
            continue
        record = '/home/huangdehao/github_projects/Proj2-ml/ori_data/training2017/' + filename
        # Read waveform samples (input is in WFDB-MAT format)
        mat_data = scipy.io.loadmat(record + ".mat")
        samples = mat_data['val']
        samples = samples[0, :]
        # preprocessing
        # TODO: Use given function to preprocess data
        new_samples = sample_preprocessing(samples)
        # Save data
        save_filename = os.path.join(test_dir, filename + '.mat')
        scipy.io.savemat(save_filename, {'value': new_samples, 'label': label})
        save_filename = os.path.join(test_without_preprocess_dir, filename + '.mat')
        scipy.io.savemat(save_filename, {'value': samples, 'label': label})
        print(save_filename)
