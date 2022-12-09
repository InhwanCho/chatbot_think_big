# 씽크빅 챗봇 프로젝트입니다.(KoGPT2,SBRT 사용)

## Bert 구조

![Bert input representation](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbABsUL%2FbtqzmTU7OLm%2FYwK6JLhNfTYvxkiFzkfkCK%2Fimg.png)

### Token Embeddings

- Word Piece 임베딩 방식 사용
- 자주 등장하면서 가장 긴 길이의 sub-word를 하나의 단위로 생성
- 즉, 자주 등장하는 sub-word은 그 자체가 단위가 되고, 자주 등장하지 않는 단어(rare word)는 sub-word로 쪼개짐
- 기존 워드 임베딩 방법은 Out-of-vocabulary (OOV) 문제가 존재하며, 희귀 단어, 이름, 숫자나 단어장에 없는 단어에 대한 학습, 번역에 어려움이 있음
- Word Piece 임베딩은 모든 언어에 적용 가능하며, sub-word 단위로 단어를 분절하므로 OOV 처리에 효과적이고 정확도 상승효과도 있음

### Sentence Embeddings

- BERT는 두 개의 문장을 문장 구분자([SEP])와 함께 결합
- 입력 길이의 제한으로 두 문장은 합쳐서 512 subword 이하로 제한
- 입력의 길이가 길어질수록 학습시간은 제곱으로 증가하기 때문에 적절한 입력 길이 설정 필요
- 한국어는 보통 평균 20 subword로 구성되고 99%가 60 subword를 넘지 않기 때문에 입력 길이를 두 문장이 합쳐 128로 해도 충분
- 간혹 긴 문장이 있으므로 우선 입력 길이 128로 제한하고 학습한 후, 128보다 긴 입력들을 모아 마지막에 따로 추가 학습하는 방식을 사용

### Position Embedding

- BERT는 저자의 이전 논문인 Transformer 모델을 착용
- Transformer은 주로 사용하는 CNN, RNN 모델을 사용하지 않고 Self-Attention 모델을 사용
- Self-Attention은 입력의 위치에 대해 고려하지 못하므로 입력 토큰의 위치 정보가 필요
- Transformer 에서는 Sinusoid 함수를 이용한 Positional encoding을 사용하였고, BERT에서는 이를 변형하여 Position encoding을 사용
- Position encoding은 단순하게 Token 순서대로 0, 1, 2, ...와 같이 순서대로 인코딩

### 임베딩 취합

- BERT는 위에서 소개한 3가지의 입력 임베딩(Token, Segment, Position 임베딩)을 취합하여 하나의 임베딩 값으로 생성
- 임베딩의 합에 Layer Normalization과 Dropout을 적용하여 입력으로 사용

![파인튜닝 구조](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbg5SlP%2FbtqzntBU7Uj%2FKHWiKI4zKgb8FqLzAYAusK%2Fimg.png)

### MLM(Masked Language Model)

![학습 방법](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLMyXN%2Fbtqzl4Ql7sH%2FykzRZNWkc6rcb8ffU5Nrm1%2Fimg.png)

#### 코랩 환경에서 챗봇 실행 방법

1.해당 파일을 코랩에서 실행합니다.

`chatbot_think_big/thinkbig_stuff/최종_실행파일/Thinkbig_chatbot(colab).ipynb`

2.런타임 유형을 `GPU`로 세팅 후 실행하면 됩니다.

3.코랩 환경에서 챗봇 실행 동영상입니다.

[![코랩 챗봇 구동 영상](https://user-images.githubusercontent.com/111936229/206372581-a6da8be0-91fa-41d9-b28c-b7574ca9d0af.png)](https://youtu.be/HHClT36nYT8)

- 버전 문제로 로컬 환경에서는 계속 오류가 나네요
- 실행 시 에러가 날때도 있고 아닐 때도 있긴한데 이유를 모르겠습니다.
- 1epch당 약 3시간이 걸리므로 유료버전에서 돌려보는 것을 추천(아니면 중간에 끊길 확률이 높습니다.)

- 참고 : <[KoGPT2-chatbo]>
- 참고 : <[ebbn]>
- 참고 : <[NLP-kr]>
- 참고 : <[이수한컴퓨터연구소]>

[KoGPT2-chatbo]: ttps://github.com/haven-jeon/KoGPT2-chatbo
[ebbn] : ttps://ebbnflow.tistory.com/151
[NLP-kr] : ttps://github.com/NLP-kr/tensorflow-ml-nlp-tf2
[이수한컴퓨터연구소] : ttps://www.youtube.com/watch?v=LEtLfx1dS7Q

- BERT는 위에서 소개한 3가지의 입력 임베딩(Token, Segment, Position 임베딩)을 취합하여 하나의 임베딩 값으로 생성
- 임베딩의 합에 Layer Normalization과 Dropout을 적용하여 입력으로 사용
