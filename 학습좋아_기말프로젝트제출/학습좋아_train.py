import wave
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import joblib   # model 저장
import os
from sklearn import preprocessing  # 데이터 스케일링


# train raw 음성 데이터를 train wav 음성 데이터로 변환
# https://docs.python.org/3/library/wave.html
# https://stackoverflow.com/questions/58661690/how-can-i-convert-a-raw-data-file-of-audio-in-wav-with-python
with open("fmcc_train.ctl", 'r') as file:
    train_file_name_list = file.readlines()
i = 0
for train_file_name in train_file_name_list:
    train_file_name = train_file_name.strip() # \n 제거
    train_raw_path = "raw16k/train/" + train_file_name + ".raw"
    train_wav_path = "raw16k/train/" + train_file_name + ".wav"
    print(i, train_file_name)
    with open(train_raw_path, "rb") as inp_f:
        data = inp_f.read()
        with wave.open(train_wav_path, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)
            out_f.setframerate(16000)
            out_f.writeframesraw(data)
    i += 1


# train_wav 음성 데이터 mfcc
with open("fmcc_train.ctl", 'r') as file:
    train_file_name_list = file.readlines()
train_mfccs = []
i = 0
for train_file_name in train_file_name_list:
    # filename
    train_file_name = train_file_name.strip() # \n 제거
    train_wav_path = "raw16k/train/" + train_file_name + ".wav"
    print(i, train_file_name)
    # mfcc
    y, sr = librosa.load(train_wav_path, sr=16000)  # https://librosa.org/doc/0.10.0/generated/librosa.load.html
    y = librosa.util.fix_length(data=y, size=40000) # https://librosa.org/doc/0.10.0/generated/librosa.util.fix_length.html # 10000개의 train_wav의 data 중 가장 큰 size가 39040
    y = librosa.effects.preemphasis(y=y, coef=0.97) # https://librosa.org/doc/0.10.0/generated/librosa.effects.preemphasis.html
    y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=201,fmax=800, fmin=10)
    M = librosa.feature.mfcc(S=librosa.power_to_db(y), n_mfcc=39, n_fft=400, hop_length=160, dct_type=3, lifter=23)
    M = preprocessing.StandardScaler().fit_transform(M) # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    deltas = librosa.feature.delta(M, width=5)
    double_deltas = librosa.feature.delta(deltas, width=5, order=2)
    M = np.concatenate([M, deltas, double_deltas], axis=0)
    train_mfccs.append(M.T)
    i+=1
# npy 파일로 저장
train_mfcc_file_name='raw16k/학습좋아_train_features'
np.save(train_mfcc_file_name, train_mfccs)  # (10000, n_mfcc, 음성길이)


# GaussianMixture model 생성 및 저장
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
train_data = np.load("raw16k/학습좋아_train_features.npy")
# female 학습
print("female 학습 시작")
gmm_female = BayesianGaussianMixture(n_init=5, random_state=0)
train_data_female = train_data[0:5000]
train_data_female = np.reshape(train_data_female, (len(train_data_female) * len(train_data_female[0]), len(train_data_female[0][0])))
gmm_female.fit(train_data_female)
joblib.dump(gmm_female, "raw16k/학습좋아_gmm_female.pkl")
print("female 학습 끝")
print("raw16k/학습좋아_gmm_female.pkl 생성 끝")
# male 학습
print("male 학습 시작")
gmm_male = BayesianGaussianMixture(n_init=5, random_state=0)
train_data_male = train_data[5000:10000]
train_data_male = np.reshape(train_data_male, (len(train_data_male) * len(train_data_male[0]), len(train_data_male[0][0])))
gmm_male.fit(train_data_male)
joblib.dump(gmm_male, "raw16k/학습좋아_gmm_male.pkl")
print("male 학습 끝")
print("raw16k/학습좋아_gmm_male.pkl 생성 끝")









