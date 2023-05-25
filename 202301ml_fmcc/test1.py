import numpy as np
import librosa.feature.spectral
import soundfile as sf
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# import Gaussian_Mixture as gm
# from math import sqrt, log, exp, pi
# from random import uniform

dir(librosa)
f = open("fmcc_train.ctl", "rb")
lines = f.readlines()
i = 0
data =[]
mfs_m=[]
init_mu=[]
init_cov=[]
for line in lines:
    new = str(line)[2:-5]
    fid = open("raw16k/train/" + new + ".raw", "rb")
    Img,sr = sf.read(fid,samplerate=16000,channels=1,subtype='PCM_16')
    print(Img)

    print(Img.shape)

    mf = librosa.feature.spectral.mfcc(y=Img,sr=16000,n_mfcc=100)
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    padded_mfcc = pad2d(mf, 40)
    print(padded_mfcc.shape)
    reshaped_mfcc = padded_mfcc.reshape(1, -1)
    reshaped_mfcc = np.squeeze(reshaped_mfcc)
    print(reshaped_mfcc.shape)
    data.append(reshaped_mfcc)

    # if i > 5:
    #     break
    # i += 1
#print(np.mean(mfs))
#print(np.linalg.eig(mfs))
# labels, _, _ = gaussian_mixture_clustering(mfs, n_components=5, init_mu=np.mean(mfs,axis=0), init_cov_mat=np.linalg.eig(mfs), init_weights=None,
#                                            epsilon=1e-4, max_iter=20, random_state=100)
# my_gmm = numpy_ml.gmm.GMM
# my_gmm1 = numpy_ml.gmm.GMM._initialize_params(my_gmm,mfs)
#my_gmm = main.gaussian_mixture_clustering(mfs,n_components=2)
#init_cov_mat = np.cov(mfs)
data = np.array(data)
print(data)
print(data.shape)
gmm = GaussianMixture(n_components=2)
gmm.fit(data) # GMM 클러스터링 수행
labels = gmm.predict(data) # 최종 클러스터 라벨링

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.scatter(data[:,0], data[:,1], c=labels)
plt.show()
