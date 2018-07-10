# NLP Tutorial with Deep Learning using tensorflow

<br />

## 1. Install requirements

```
$ pip install -r requirements.txt
```

<br />

## 2. [01] sentiment_analysis

네이버 영화 평점 데이터를 이용하여 여러 딥러닝 모델을 비교해 볼 수 있습니다.<br />
다양한 컴퓨터 사양으로 테스트 해 볼 수 있도록 데이터셋을 크기에 따라 세 가지를 두었습니다.<br /><br />

### Contents
1. [Data Process]
2. [Logistic Regression](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B01%5D%20logistic_regression.ipynb)
3. [Feed Forward Neural Network](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B02%5D%20three_layer_net.ipynb)
4. [RNN](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B03%5D%20RNN.ipynb): with rnn tensorflow API explanation
5. [LSTM](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B04%5D%20LSTM.ipynb)
6. [CNN](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B05%5D%20CNN.ipynb)
7. [word2vec](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B08%5D%20word2vec.ipynb)
8. [doc2vec](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/%5B09%5D%20doc2vec.ipynb)
9. Every model is defined at [models](https://github.com/Bricoler/nlp-tensorflow/blob/master/%5B01%5D%20sentiment_analysis/models.py)

<br />

## 3. [02] nlp_application

딥러닝 모델을 이용하여 사용자의 입력을 받아 [감성분석, 삼행시만들기, 대화하기]를 테스트 할 수 있습니다.<br />
훈련된 모델이 같이 들어있어 train 없이 바로 test 할 수 있으며, gpu 없이 train 할 시, 시간이 오래 걸릴 수 있습니다.<br /><br />

### train

```
$ python train.py
```

### implementation

```
$ python test.py
```

<br />

## 4. Notice
1. [02] nlp_application/[01] Sentiment 의 경우, models 폴더 안 model-31000.zip 파일의 압축을 풀어주어야 train 없이 바로 test 할 수 있습니다.<br />
### ubuntu/mac: how to unzip in command line
```
$ zip -FF model-31000.zip --out model-31000-full.zip
$ unzip model-31000-full.zip
```

<br />

2. 모든 데이터는 한국어로 이루어져 있습니다.<br /><br />

3. konlpy 설치는 <href>http://konlpy.org/en/v0.4.4/install/</href> 를 참조하세요.<br /><br />
