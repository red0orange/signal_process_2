import os
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

import neurokit2 as nk

from Pan_tompkins_algorithm import Pan_tompkins, HeartRateMaintainer


def preprocess(data):
    # nyq_rate = 300 / 2.0
    # a = signal.firwin(512,60.0/nyq_rate)
    # data = signal.filtfilt(a,1,data)
    # data = signal.resample_poly(data,120,300)
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    data = (data-mu)/sigma
    return data


def process(data_root, rows):
    sample_rate = 300

    Xs = []

    for ii, row in enumerate(rows):
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
        record = os.path.join(data_root, filename)
        # Read waveform samples (input is in WFDB-MAT format)
        mat_data = scipy.io.loadmat(record + ".mat")
        samples = mat_data['val']
        samples = samples[0, :]

        # Preprocessing
        signal = preprocess(samples)

        try:
            waves_peak, _ = nk.ecg.ecg_process(signal, sampling_rate=sample_rate, method="neurokit")

            clean_signal = waves_peak['ECG_Clean']
            rates = waves_peak['ECG_Rate']
            R_peaks = np.where(waves_peak['ECG_R_Peaks'])[0]
            P_peaks = np.where(waves_peak['ECG_P_Peaks'])[0]
            T_peaks = np.where(waves_peak['ECG_T_Peaks'])[0]
            Q_peaks = np.where(waves_peak['ECG_Q_Peaks'])[0]
            S_peaks = np.where(waves_peak['ECG_S_Peaks'])[0]
            Xs.append([label, np.max(rates), np.min(rates), len(P_peaks), len(T_peaks), len(Q_peaks), len(S_peaks)])
        except BaseException:
            print("Error")

    # for X in Xs:
    #     result.append(X)
    return Xs


if __name__ == "__main__":
    num_process = 80
    data_root = "/home/dehao/github_projects/signal_process_2/ori_data/training2017"
    # data_path = "/home/dehao/github_projects/signal_process_2/data/train_data.csv"
    # save_path = "/home/dehao/github_projects/signal_process_2/ml_data/train.npy"
    data_path = "/home/dehao/github_projects/signal_process_2/data/test_data.csv"
    save_path = "/home/dehao/github_projects/signal_process_2/ml_data/test.npy"
    
    rows = []
    with open(data_path, "r") as f:
        num_samples = len(f.readlines())
    with open(data_path, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            rows.append(row)

    each_process_sample_num = num_samples // num_process
    results = []
    pool = multiprocessing.Pool(num_process)
    for i in range(num_process):
        start_index = i * each_process_sample_num
        end_index = (i+1) * each_process_sample_num
        results.append(pool.apply_async(func=process, args=(data_root, rows[start_index:end_index])))
    pool.close()
    pool.join()
    Xs = []
    for result in results:
        Xs.extend(result.get())
    
    np.save(save_path, Xs)