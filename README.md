# Software-Convergence-Capstone (2022-1)
## Dynamic Video Summarization with attentive mechanism
소프트웨어융합학과 17학번 강성문
> Library : Tensorflow, Scikit-learn, Numpy, Pandas, Keras, h5py, etc 
> 
> Environment : Windows 10, Python 3.8
## 1. Overview
### 1.1. 과제 선정 배경
- 스마트폰 카메라 및 다양한 촬영 장비의 등장과 동영상 업로드 인터넷 플랫폼(YouTube, 
etc)가 등장함에 따라 매 시간마다 업로드되는 동영상의 양은 매우 방대하다. 하지만, 모든 영상을 
처음부터 끝까지 시청하는 것은 굉장히 큰 시간이 소요된다. 이에 따라, 수많은 동영상의 
중요 부분을 요약하여 시청자들로 하여금 효율적으로 접근할 수 있게 하는 기술이 중요해지는 상황이다.

- 동영상 요약이라는 주제의 과제는 수년간 연구되어 왔지만, 개인별로 하이라이트를 뽑아내는 기준의 다양성과 비정형 데이터의 한계, 데이터셋의 부족과 동영상 도메인의 다양성으로 완벽한 요약을 수행하기 어렵다는 특징을 지닙니다.
### 1.2. 과제 주요 내용
- 동영상 요약은 크게 주요 장면 단위만 추출하는 스토리보드의 형식과, 음성과 모션을 살려 요약하는 Skimming 방식으로 나눌 수 있다. 본 과제에서는 후자의 방식을 채택하여 
frame간의 temporal적 특성에 초점을 맞추어 짧은 요약 영상을 생성하는 것을 목표로 한다. 

- 요약을 원하는 동영상을 입력으로 넣으면, Frame단위로 변환하여, CNN(GoogleNet, ResNet)
의 pooling ouput 구조를 이용해 Frame별 Feature를 이끌어 내어 input 데이터로 활용한다. 이들을 encoder-decoder 구조를 거쳐 frame-level importance에서 Kernel Temporal 
segmentation을 이용해 최종 Shot-level importance를 산출하여, 동영상 요약의 최대 길이를 
넘지 않는 선(전체 길이의 15%)에서 가장 유의미한 동영상을 만들어내는 것이 목표이다. 
- 요약이 완료된 동영상은 실제 사람이 중요하다고 평가한 Frame들과 얼마나 Overlap되는지를 이용해 성능을 측정하고, Precision, Recall을 측정하여 F-Score를 최종 평가 지표로써 활용한다. 
- 데이터셋으로는 동영상 요약 프로젝트에서 자주 언급되며 유저에 의해 생성된 하이
라이트 Label을 제공하는 TVSum, SumMe 데이터를 사용한다.
### 1.3. 과제 목표
- 기존 연구에서 나타난 성능(TVsum F-Score : 0.60, Summe F-Score : 0.43)을 기준으로 하며, 알고리즘 개선을 통해 해당 성능을 넘기는 것을 목표로 한다.

--------------------------
## 2. Dataset
### 2.1. 데이터 수집
TVsum Dataset : https://github.com/yalesong/tvsum 
<br/><br/>
Summe Dataset : https://gyglim.github.io/me/vsum/index.html

--------------------------

### 2.2. 데이터 전처리
#### 2.2.1. Frame-level Feautre Extraction
- 동영상 데이터를 Frame별로 딥러닝 모델 input에 맞는 형태로 변환해주는 과정을 거친다.

- Feature input을 추출하기 위해 Imagenet Pretrained CNN 모델들을 사용하여 가장 성능이 우수한 모델을 선정한다. 각각의 모델에서 Fully-connected layer 이전 단계까지의 결과값을 사용한다. 본 프로젝트에서 실험한 모델은 다음과 같다. 각각의 모델에 따른 성능 비교는 Result에 함께 서술한다.

``` python
net = models.googlenet(pretrained=True).float()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2]) # pool5 layer
```

``` python
net = models.inception_v3(pretrained=True).float()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2]) # pool5 layer
```

``` python
net = models.Resnext101_32x8d(pretrained=True).float()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-1]) # pool5 layer
```



&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; → Googlenet(baseline) / Inception v3 / Resnext101_32x8d 
<br/><br/>
![image](https://user-images.githubusercontent.com/65675861/173499115-4036b421-df72-4b3a-a8cc-2f6887f013cb.png)
<p align="center">
  Googlenet Example
</p>

###

#### 2.2.2. Change point Detection for Segmentation of Video
- 모델에서 계산된 Frame-level importance를 Shot-level importance로 변환하기 위해서 필요한 Change point를 계산한다.

- Video Segmentation 기법으로는 Frame variance를 활용하는 KTS(Kerenel Temporal Segmentation) 기법을 활용한다.

- KTS code : https://github.com/TatsuyaShirakawa/KTS

-----------------------------
### 2.3. Create h5 file

|Key|Description|
|----|-----|
|feature|2D-array with shape (320, 1024) contains feature vectors representing video frames. Each video frame can be represented by a feature vector (containing some semantic meanings), extracted by a pretrained convolutional neural network (e.g. GoogLeNet). It is used in traning, test and inference time. Trained for the image classification task.|
|label|1D-array with shape (n_frames), each row is a binary vector (used for test) contains multiple key-clips given by human annotators and we need to compare our machine summary with each one of the user summaries.|
|change_points|2D-array with shape (num_segments, 2), each row stores indices of a segment corresponds to shot transitions, which are obtained by temporal segmentation approaches that segment a video into disjoint shots num_segments is number of total segments a video is cut into.|
|n_frame_per_seg|1D-array with shape (num_segments), indicates number of frames in each segment.|
|fps|frames per second of the original video|
|video_name|original video name|

---------------------------
## 3. Model
> 기존 연구 중에서 Zhong Ji et al., Video Summarization with Attention-Based Encoder-Decoder Networks, Xi’an Institute of Optics and Precision Mechanics, 2018 논문을 참고하여 모델을 수립한다. 
### 3.1. Model Structure
![image](https://user-images.githubusercontent.com/65675861/173501754-2d2adec4-f4af-4d25-80a8-368f27dcc86d.png)
####
- Video에서 Pretrained CNN모델을 통해 Frame-level Feature를 추출

- BiLSTM Encoder -> LSTM Decoder(with attention layer) -> Frame-level importance -> Shot-level importance(Segment-level)

- 0/1 knapsack alogrithm (baseline), Fractional knapsack algorithm (proposed)를 이용해 비디오 길이의 15%를 넘지 않는 선에서 Segment
---------------
### 3.2. LSTM & BiLSTM
![image](https://user-images.githubusercontent.com/65675861/145566421-4b91e92f-1843-4565-9093-3b3bcf460cc7.png)
![image](https://user-images.githubusercontent.com/65675861/173503132-28e50df2-5a16-4a1e-901d-54349d9898f8.png)

####
- 시계열 데이터에서 자주 사용되는 딥러닝 모델로, RNN과 유사하지만 Neural Network Layer 1개의 층 대신 4개의 layer가 존재

- BiLSTM을 통해 양방향 LSTM 구조를 갖추게 함으로써, 비디오 내 순방향, 역방향의 정보를 더욱 잘 보존하며 학습하도록 함
``` python
encoder_BidirectionalLSTM = Bidirectional(LSTM(128, return_sequences = True, return_state = True))
```
-----------------------
### 3.3. Attention Layer
![image](https://user-images.githubusercontent.com/65675861/173503321-4cd6946e-1dc3-4132-9aef-8b126992f57a.png)
####
- Video내 모든 Frame을 동일한 가중치로 다루는 것을 막아, Encoder 결과값을 이용하여 가중치를 부여하는 알고리즘

``` python
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM(encoder_inputs)
sh = Concatenate()([fh, bh])
ch = Concatenate()([fc, bc])
encoder_states = [sh, ch]

decoder_LSTM = LSTM(256, return_sequences = True)
decoder_out = decoder_LSTM(encoder_out, initial_state = encoder_states)

attn_layer = Attention(name="Attention_Layer")

attn_out =  attn_layer([encoder_out, decoder_out])

decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_out, attn_out])
```
--------------
### 3.4. Train
``` python
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state=0) # To make same environment for every comparison
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.15)
batch_size = 5 ; epoch = 5
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
```
--------------
### 3.5. Knapsack Algorithm in summarization
- 대부분의 비디오 요약 연구에서 수행되었던 알고리즘은 Dynamic Programming 기반의 0/1 Knapsack Algorithm 이다. 하지만, 0/1 Knapsack Algorithm을 적용하면 동영상 요약 허용 길이를 가득 채우지 못하게 되는 경우가 빈번히 발생하는 점에 착안하여 Greedy Alogrithm 기반의 Fractional knapsack 알고리즘을 제안한다. 

- 또한, 0/1 Knapsack Algorithm에 최선의 해결책을 탐색해 나가는 Genetic Algorithm을 적용하여 이들의 성능과 정확도를 비교한다. DP 기반의 0/1 knapsack의 경우 현재 최적의 방법으로 여겨지므로 정답인 100%로 가정하고, genetic algorithm의 value 도출 정도(fitness)를 평가한다.


|Algorithm|Time Complexity|Accuracy|etc|
|----|-----|-----|----|
|0/1 Knapsack Algorithm(DP)|O(nW)|100%|n : segment 개수 / W : 0.15 x length|
|0/1 Knapsack Algorithm(GA)|O(Gnm)|91.6%|G : generation 횟수 / n : segment 개수 / m : 1회 생성 시 만들어지는 유전자 개수|
|**Fractional Knapsack Algorithm(Greedy)**|**O(nlogn)**|**100%**|**n : segment 개수**|

- Genetic Algorithm의 경우 Segment의 개수가 커지는 비디오의 경우 정확도와 시간 측면에서 떨어진다. 일반적으로 시간복잡도 측면에서 Fractional knapsack algorithm이 더 빠르지만, W 값이 작다면 0/1 Knapsack이 더 빠르게 수행되는 경우도 있다는 결론을 도출했다.


--------------
## 4. Result
### 4.1. TVsum 
|CNN Model|Knapsack Algorithm|F-score|Note|
|----|-----|---|----|
|Googlenet|0/1 knapsack(DP)|0.60|baseline|
|Googlenet|0/1 knapsack(GA)|0.42||
|Googlenet|Fractional knapsack|0.61||
|Inception_v3|0/1 knapsack(DP)|0.59||
|Inception_v3|0/1 knapsack(GA)|0.40||
|Inception_v3|Fractional knapsack|0.60||
|Resnext101_32x8d|0/1 knapsack(DP)|0.61||
|Resnext101_32x8d|0/1 knapsack(GA)|0.44||
|**Resnext101_32x8d**|**Fractional knapsack**|**0.62**||

### 4.2. Summe
|CNN Model|Knapsack Algorithm|F-score|Note|
|----|-----|---|----|
|Googlenet|0/1 knapsack(DP)|0.42|baseline|
|Googlenet|0/1 knapsack(GA)|0.25||
|Googlenet|Fractional knapsack|0.43||
|Inception_v3|0/1 knapsack(DP)|0.40||
|Inception_v3|0/1 knapsack(GA)|0.26||
|Inception_v3|Fractional knapsack|0.42||
|Resnext101_32x8d|0/1 knapsack(DP)|0.43||
|Resnext101_32x8d|0/1 knapsack(GA)|0.31||
|**Resnext101_32x8d**|**Fractional knapsack**|**0.45**||


---------------------
## 5. Conclusion
### 5.1. 결론 및 제언
- 동적 동영상 요약이라는 프로젝트는 동영상 데이터의 특성상 완벽한 예측이 불가능하다는 점을 느꼈다. 특히, 도메인이 다른 영역의 비디오들간에 하나의 동일한 학습된 모델을 이용하여 요약 지점을 이끌어내는 것도 현재는 의문이 든다. 하지만, 동영상 요약 과제에서 활용되는 데이터셋이 충분해지고 다양한 분야와 길이의 동영상을 학습하는 딥러닝 모델 구현이 가능하다면 충분히 발전시킬 수 있는 과제라고 생각한다. 

- 마지막으로, KTS 코드에 대한 소개와 자료가 부족하여 Change point를 Detection하는 것에 어려움이 있었는데, 보다 정교하게 Segmentation하여 성능을 개선시키는 방안을 모색하고 프로젝트를 마무리하고자 한다.

### 5.2. 활용 방안
- 동영상 시청자들은 전체 영상을 시청하며 많은 시간을 소비하지 않게 되며, 핵심 내용만을 빠르게 
파악할 수 있다.

- 편집된 동영상을 업로드하는 많은 동영상 플랫폼 사용자(Youtube, etc)들의 동영상 편집 시간을 
크게 단축시킬 수 있다.

---------------
## 6. Reference 
Zhong Ji et al., Video Summarization with Attention-Based Encoder-Decoder Networks, Xi’an Institute of Optics and Precision Mechanics, 2018.
