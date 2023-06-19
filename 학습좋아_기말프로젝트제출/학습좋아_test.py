print("input : " , end="")

import wave
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import joblib   # model 저장
import os
from sklearn import preprocessing  # 데이터 스케일링

inputFile = input()
if inputFile == "fmcc_eval.ctl":
    outputFile = "학습좋아_eval_results.txt"
elif inputFile == "fmcc_test900.ctl":
    outputFile = "학습좋아_test_results.txt"
else:
    print("please input \"fmcc_eval.ctl\" or \"fmcc_test900.ctl\"")
    exit(0)


# test raw 음성 데이터를 test wav 음성 데이터로 변환
# https://docs.python.org/3/library/wave.html
# https://stackoverflow.com/questions/58661690/how-can-i-convert-a-raw-data-file-of-audio-in-wav-with-python
with open(inputFile, 'r') as file:
    test_file_name_list = file.readlines()
i = 0
for test_file_name in test_file_name_list:
    test_file_name = test_file_name.strip() # \n 제거
    if inputFile == "fmcc_eval.ctl":
        test_raw_path = "raw16k/eval/" + test_file_name + ".raw"
        test_wav_path = "raw16k/eval/" + test_file_name + ".wav"
    elif inputFile == "fmcc_test900.ctl":
        test_raw_path = "raw16k/test/" + test_file_name + ".raw"
        test_wav_path = "raw16k/test/" + test_file_name + ".wav"
    print(i, test_file_name)
    with open(test_raw_path, "rb") as inp_f:
        data = inp_f.read()
        with wave.open(test_wav_path, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)
            out_f.setframerate(16000)
            out_f.writeframesraw(data)
    i+=1


# test_wav 음성 데이터 mfcc  +   test_result.txt에 결과 쓰기  +   테스트 결과 확인
gmm_female = joblib.load("raw16k/학습좋아_gmm_female.pkl")
gmm_male = joblib.load("raw16k/학습좋아_gmm_male.pkl")

with open(inputFile, 'r') as file:
    test_file_name_list = file.readlines()
i = 0
if os.path.isfile(outputFile):
    os.remove(outputFile)
with open(outputFile, 'a') as file:
    for test_file_name in test_file_name_list:
        # filename
        test_file_name = test_file_name.strip() # \n 제거
        if inputFile == "fmcc_eval.ctl":
            test_raw_path = "raw16k/eval/" + test_file_name + ".raw"
            test_wav_path = "raw16k/eval/" + test_file_name + ".wav"
        elif inputFile == "fmcc_test900.ctl":
            test_raw_path = "raw16k/test/" + test_file_name + ".raw"
            test_wav_path = "raw16k/test/" + test_file_name + ".wav"
        print(i, test_file_name)
        # mfcc
        y, sr = librosa.load(test_wav_path, sr=16000)  # https://librosa.org/doc/0.10.0/generated/librosa.load.html
        y = librosa.util.fix_length(data=y, size=40000) # https://librosa.org/doc/0.10.0/generated/librosa.util.fix_length.html # 10000개의 train_wav의 data 중 가장 큰 size가 39040
        y = librosa.effects.preemphasis(y=y, coef=0.97) # https://librosa.org/doc/0.10.0/generated/librosa.effects.preemphasis.html
        y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=201,fmax=800, fmin=10)
        M = librosa.feature.mfcc(S=librosa.power_to_db(y), n_mfcc=39, n_fft=400, hop_length=160, dct_type=3, lifter=23)
        M = preprocessing.StandardScaler().fit_transform(M) # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        deltas = librosa.feature.delta(M, width=5)
        double_deltas = librosa.feature.delta(deltas, width=5, order=2)
        M = np.concatenate([M, deltas, double_deltas], axis=0)
        M = M.T
        # score
        f_hood = np.array(gmm_female.score(M)).sum()
        m_hood = np.array(gmm_male.score(M)).sum()
        # write
        file.writelines(test_raw_path)
        if f_hood >= m_hood: file.writelines(' feml\n')
        else: file.writelines(' male\n')
        i+=1

print(outputFile, "생성 끝")
