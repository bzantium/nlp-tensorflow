# NLP Tutorial with Deep Learning using tensorflow

<br />

## 1. install requirements

```
pip install -r requirements.txt
```

<br />

## 2. [01] sentiment_analysis

네이버 영화 평점 데이터를 이용하여 여러 딥러닝 모델을 비교해 볼 수 있습니다.<br />
다양한 컴퓨터 사양으로 테스트 해 볼 수 있도록 데이터셋을 크기에 따라 세 가지를 두었습니다.<br /><br />
You can compare various deep learning models using naver movie review dataset.<br />
In order to make users able to test within their computer's capability, I uploaded three dataset with different sizes.

<br />

## 3. [02] nlp_application

딥러닝 모델을 이용하여 사용자의 입력을 받아 [감성분석, 삼행시만들기, 대화하기]를 테스트 할 수 있습니다.<br />
훈련된 모델이 같이 들어있어 train 없이 바로 test 할 수 있으며, gpu 없이 train 할 시, 시간이 오래 걸릴 수 있습니다.<br /><br />
You can test models for [Sentiment Analysis, SamHangSi, Conversation] given user's input trained using deep learning.<br />
Since this repo includes trained models, you can test the models without training by yourself.<br />
If you train the models without gpu, it can take quite long time.<br />

### train

```
python train.py
```

### implementation

```
python test.py
```

<br />

## 4. notice
1. [01] Sentiment 의 경우, models 폴더 안 model-31000.zip 파일의 압축을 풀어주어야 train 없이 바로 test 할 수 있습니다.<br /><br />
For [01] Sentiment, you must unzip model-31000.zip in folder 'models' to test the model without training.<br /><br />

2. 모든 데이터는 한국어로 이루어져 있습니다.<br /><br />
Every dataset is written in Korean.<br /><br />

3. konlpy 설치는 <href>http://konlpy.org/en/v0.4.4/install/</href> 를 참조하세요.<br /><br />
For installation 'konlpy', please refer to <href>http://konlpy.org/en/v0.4.4/install/</href>
