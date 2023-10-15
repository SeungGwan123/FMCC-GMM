# FMCC-GMM (남성-여성 음성 분류)
<div align="center">
<pre>
  이민규 이승관
인천대학교
mingyu@inu,ac.kr, skwan123@naver.com
FMCC-GMM
LEEMINGYU LEESEUNGGWAN
IncheonNationalUniversity
 
머 신 스 펙 & 소프트웨어
  OS – Window11 / CPU - 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz / RAM - 32GB
  GPU – NVIDIA GeForce RTX 3060 Laptop GPU / SSD – NVMe PC711 NVMe SK hynix 1TB
  Software – IntelliJ IDEA 2023.1.2. & Jupyter Notebook 1.0.0
</pre>
</div>

##  I. 서 론
<pre>
음성 인식은 인간의 음성을 컴퓨터가 이해하고 처리하는 분야에서 중요한 역할을 합니다. 음성 인식 기술은 음성 데이터를 분석하여 
텍스트로 변환하거나, 음성 명령을 이해하고 실행하는 등의 다양한 응용 프로그램에 사용됩니다. 특히, 음성 데이터를 성별에 따라 
분류하는 작업은 음성 인식 분야에서의 중요한 주제 중 하나입니다. 본 보고서에서는 GMM (Gaussian Mixture Model)을 
사용하여 노이즈가 있는 남성과 여성 음성 데이터를 훈련하고, 이를 효과적으로 분류하는 방법을 탐구합니다. 이를 위해 우리는 원시 
음성 데이터를 처리하여 wav 파일로 변환하고, mfcc (Mel-frequency Cepstral Coefficients)를 사용하여 음성 
데이터에서 특징을 추출했습니다. mfcc는 음성 신호를 주파수 대역으로 분해하고, 각 대역의 주파수 성분을 나타내는 계수들을 
추출하는 기법입니다. 이렇게 추출한 특징 데이터를 GMM 모델에 학습시킴으로써 남성과 여성 음성을 구분할 수 있는 확률 모델을 
생성했습니다. 본 논문은 다음과 같이 구성된다. II장에서는 본 연구의 특징 추출의 기반이 되는 MFCC에 관하여 기술한다. 
III장에서는  가우시안 혼합 모델의 이론에 관하여 기술한다. IV장에서는 가우시안 혼합 모델을 이용한 연구결과를 기술하고, 
Ⅴ장에서 논문의 결론을 맺는다.
</pre>

##  II. MFCC

<pre>
II장에서는 본 논문에서 선택한 특징 추출 방식인 MFCC에 대해서 설명한다. MFCC는 Mel Frequency Cepstral 
Coefficients의 약자로 음성데이터를 특징벡터로 바꾸는 방식이다. 총 5개의 단계로 진행이 되는데 1단계는 Pre-emphasis로 
고주파 성분의 에너지를 강조하여 음성 신호의 고주파 성분을 강화시키는 역할을 하면서 주변 환경의 노이즈를 줄이기도 한다. 
2단계는 Windowing으로 음성 신호를 작은 조각으로 나누는 과정이다. 음성데이터의 시간적인 특성을 보존하면서 주파수 정보를 
추출하는 중요한 과정이다. 이 과정에서 핵심이 일정 부분을 겹쳐 잘라내어 연속성을 유지하는 것이다.
<img src = "image/스크린샷 2023-10-15 오후 2.39.41.png" width="600" height="200">
3단계는 Fourier transform으로 도메인의 신호를 주파수 도메인으로 변환하는 수학적인 기법으로 시간 도메인에서 관찰된 신호를 
주파수 구성 요소로 변환을 한다. 4단계는 Log Mel filter bank analysis로 주파수로 변환한 데이터를 분석하는 단계이다. 
실제 사람이 인식하는 주파수 관계를 사용하여 분석을 한다.
<img src = "image/스크린샷 2023-10-15 오후 2.45.45.png" width="600" height="200">
마지막 단계는 Discrete cosine transform으로 시간 영역에서 주파수 영역으로 신호를 변환하는 변환 기술이다. 변환된 결과에서 
주요한 주파수 성분을 추출함으로써 MFCC의 특징벡터가 만들어진다.
<img src = "image/스크린샷 2023-10-15 오후 2.45.50.png" width="600" height="200">
</pre>

## III. 베이지안 가우시안 혼합 모델(BGM) 이론
<pre>
이 장에서는 본 논문에서 학습 모델로 선택한 베이지안 가우시안 혼합 모델(BGM)과 가우시안 혼합 모델(GMM)의 이론에 대하여 
설명한다. 가우시안 확률밀도함수는, 다양한 확률분포함수 중에서 대표적으로 가장 많이 사용하는 분포이다. d차원 공간에서의 한 점인 
중심 μ와 공분산 행렬 Σ가 정해지면, d차원을 갖는 특징 벡터를 확률변수로 한 가우시안 확률밀도함수는 다음과 같이 표현할 수 있다.
<img src = "image/스크린샷 2023-10-15 오후 2.46.00.png" width="500" height="100">
GMM은 이러한 가우시안 확률밀도함수 복수 개를 이용하여 데이터의 분포를 모델링하는 방법이다. GMM의 학습 단계에서는 모델의 
파라미터를 반복적으로 업데이트하면서 데이터의 우도(likelihood)를 최대화하는 EM(Expectation-Maximization) 
알고리즘이 사용된다. GMM은 군집 수를 사전에 지정해야 하며, 파라미터의 불확실성을 고려하지 않는다는 한계가 있어, 이를 극복하기 
위해 BGM이 등장하였다. BGM은 Variational Bayesian methods (변분 베이지안 방법)을 사용하여 파라미터의 불확실성을 
모델링하고, 군집의 수를 자동으로 결정하여 클러스터링에 더 적합한 모델이 되었다. Variantional Bayesian methods는 
KL(Kullback-Leibler divergence)를 사용하여 사후 분포와 근사 분포 사이의 거리를 측정하고, 이를 최소화하는 방식으로 
근사 분포를 결정한다. 이를 위해 Variational optimization (변분적 최적화)를 사용하는데, 이는 파라미터를 조정하여 KL 
divergence를 최소화하는 방향으로 근사 분포를 업데이트한다.
</pre>

## IV. 연구결과
<pre>
본 연구에서의 최종 정확도는 95.33%가 나왔다. 일단 librosa라이브러리 중 fix_length를 사용해 음성 파일의 길이를 
동일하게 맞춰주었다. 그리고 preemphasis를 0.97로 설정했고 melspectrogram에서는 n_mels=201, fmax=800, 
fmin=10으로 설정해주었다. n_mels는 클수록 상세한 주파수 특성을 모델링 할 수 있다고 해서 가능한 크게 설정해주었다. 
그리고 남성 및 여성의 평소 말소리 주파수가 400Hz를 넘지 않으므로 fmax는 800으로 설정해주었다. MFCC에서는 
n_mfcc=39, n_fft=400, hop_length=160,dct_type=3,lifter=23으로 설정해주었다. n_mfcc는 39개로 특징을 
추출하였을 때 결과가 제일 좋았고 n_fft,hop_length는 ppt에서 주어진 값으로 맞춰주었다. dct_type은 type-1,type-2와는 
다른 계수 추출 방법을 사용함으로써 특수한 경우에 사용을 해야한다. 본 연구에서는 3으로 설정해주었을 때 정확도가 높게나왔다. 
스케일링은 StandardScaler를 사용하였다. 그리고 MFCC특징을 librosa의 delta를 사용하여 일차미분, 이차미분 값들을 
같이 특징으로 사용하였다. 모델 파라미터는 난수 고정을 위한 random_state=0으로 고정하고 n_init=5로 설정하였다. 
모델을 2개 생성하고, 하나는 남성의 음성으로만, 다른 하나는 여성의 음성으로만 학습시키는 방식으로 모델을 학습시켰다. 우리가 
사용한 모델의 총 파라미터는 2개이고 모델의 사이즈는 431KB이다. 900개의 테스트 데이터에서 각각의 데이터가 남성 모델에서의 
likelihood와 여성 모델에서의 likelihood 중 어느 것이 크게 나오는지 비교하여, label을 생성해 주었다. 아래 학습률 
곡선은 학습 데이터의 개수에 따른 정확도이다.
<img src = "image/스크린샷 2023-10-15 오후 2.46.07.png" width="500" height="200">
<img src = "image/스크린샷 2023-10-15 오후 2.46.14.png" width="500" height="200">
</pre>

## 출처
<pre>
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
https://dacon.io/codeshare/4526
https://sooftware.io/mfcc/
https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
https://zephyrus1111.tistory.com/183
https://youdaeng-com.tistory.com/5
https://ratsgo.github.io/speechbook/docs/fe/mfcc
https://github.com/SuperKogito/Voice-based-gender-recognition
https://docs.python.org/3/library/wave.html
https://stackoverflow.com/questions/58661690/how-can-i-convert-a-raw-data-file-of-audio-in-wav-with-python
</pre>
