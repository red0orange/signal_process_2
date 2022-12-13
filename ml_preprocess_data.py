import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

import biosppy
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


def features_extract(data):

    pass


def QRC_detection():
    pass


if __name__ == "__main__":


    train_root = "/home/huangdehao/github_projects/Proj2-ml/ori_data/training2017"
    test_root = "/home/huangdehao/github_projects/Proj2-ml/ori_data/sample2017"

    train_csv_path = "/home/huangdehao/github_projects/Proj2-ml/data/train_data.csv"
    test_csv_path  = "/home/huangdehao/github_projects/Proj2-ml/data/test_data.csv"

    train_save_dir = "/home/huangdehao/github_projects/Proj2-ml/ml_data/train"
    test_save_dir = "/home/huangdehao/github_projects/Proj2-ml/ml_data/test"

    # Parameters
    sample_rate = 300

    # train
    for i, (typ, data_root, csv_path, save_dir) in enumerate(zip(["train", "test"], [train_root, test_root], [train_csv_path, test_csv_path], [train_save_dir, test_save_dir])):
        csv_reader = csv.reader(open(csv_path))

        Xs = []
        Ys = []
        for ii, row in enumerate(csv_reader):
            print("cur index: {}".format(ii))
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

            # plt.plot(signal)
            # plt.savefig("test1.png")
            # plt.clf()

            # ts, filtered_signal, R_peaks, templates_ts, templates, heart_rate_ts, heart_rate = biosppy.signals.ecg.ecg(signal, sampling_rate=sample_rate, show=False)

            # _, waves_peak = nk.ecg_delineate(filtered_signal, R_peaks, sampling_rate=sample_rate, method="peak")

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

            # print(label, np.mean(rates), len(R_peaks), len(P_peaks), len(T_peaks), len(Q_peaks), len(S_peaks))
            # plt.plot(clean_signal)
            # plt.scatter(R_peaks, clean_signal[R_peaks], color = 'red', s = 50, marker= '*')
            # plt.scatter(P_peaks, clean_signal[P_peaks], color = 'blue', s = 50, marker= '*')
            # plt.scatter(T_peaks, clean_signal[T_peaks], color = 'green', s = 50, marker= '*')
            # plt.scatter(Q_peaks, clean_signal[Q_peaks], color = 'black', s = 50, marker= '*')
            # plt.scatter(S_peaks, clean_signal[S_peaks], color = 'yellow', s = 50, marker= '*')
            # plt.xlim([0, 3000])
            # plt.savefig("ttt.png")
            # pass

            # plt.plot(signal)
            # plt.scatter(r_peaks, signal[r_peaks], color = 'red', s = 50, marker= '*')
            # plt.savefig("test3.png")
            # plt.clf()

            # # Feature extract
            # QRC = Pan_tompkins(signal, sample_rate)
            # QRC_complexs = QRC.fit()
            # band_pass_signal = QRC.filtered_BandPass
            # integrated_signal = QRC.integrated_signal

            # hr_maintainer = HeartRateMaintainer(signal, sample_rate, integrated_signal, band_pass_signal)
            # R_point_xs = list(set(hr_maintainer.find_r_peaks()))
            # R_point_xs = np.array(R_point_xs)
            # R_point_xs = R_point_xs[R_point_xs > 0]
            # R_point_ys = [QRC_complexs[x] for x in R_point_xs]
            # mean_R_point_y, std_R_point_y = np.mean(R_point_ys), np.std(R_point_ys) 
            # R_point_xs = [R_point_xs[i] for i in range(len(R_point_xs)) if abs(R_point_ys[i] - mean_R_point_y) < (3*std_R_point_y)]
            # R_point_ys = [QRC_complexs[x] for x in R_point_xs]

            # # heart rate feature
            # heart_rate = len(R_point_xs) / ((len(signal) / sample_rate) / 60)

            # # 

            # # plot            
            # plt.plot(QRC_complexs)
            # plt.scatter(R_point_xs, QRC_complexs[R_point_xs], color = 'red', s = 50, marker= '*')
            # plt.savefig("test2.png")
            # plt.clf()

            # plt.plot(signal)
            # plt.scatter(R_point_xs, signal[R_point_xs], color = 'red', s = 50, marker= '*')
            # plt.savefig("test3.png")
            # plt.clf()

            # Save data
            # save_filename = os.path.join(save_dir, filename + '.mat')
            # scipy.io.savemat(save_filename, {'value': signal, 'label': label})
        save_filename = os.path.join(save_dir, filename + '.npy')
        np.save(save_filename, Xs)


        pass