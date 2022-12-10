# 씽크빅 챗봇 프로젝트입니다.(KoGPT2,SBERT 사용)

## Bert 구조

- Fine Tunning(파인튜닝)을 하기 위해서는 기존의 학습되기 전의 데이터 타입으로 바꿔 줄 필요가 있습니다.
- 즉 파인튜닝을 위해서는 Bert방식의 데이터 정제가 필요합니다.
- Bert는 아래의 그림과 같이 3가지의 입력 임베딩(Token, Segment, Position 임베딩)의 합으로 구성되어 학습된 모델입니다.

![Bert input representation](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbABsUL%2FbtqzmTU7OLm%2FYwK6JLhNfTYvxkiFzkfkCK%2Fimg.png)

### Token Embeddings

- Word Piece 임베딩 방식 사용
- 자주 등장하면서 가장 긴 길이의 sub-word를 하나의 단위로 생성
- 즉, 자주 등장하는 sub-word은 그 자체가 단위가 되고, 자주 등장하지 않는 단어(rare word)는 sub-word로 쪼개짐
- 기존 워드 임베딩 방법은 Out-of-vocabulary (OOV) 문제가 존재하며, 희귀 단어, 이름, 숫자나 단어장에 없는 단어에 대한 학습, 번역에 어려움이 있음
- Word Piece 임베딩은 모든 언어에 적용 가능하며, sub-word 단위로 단어를 분절하므로 OOV 처리에 효과적이고 정확도 상승효과도 있음

### Sentence Embeddings

- BERT는 두 개의 문장을 문장 구분자([SEP],스페셜 토큰)와 함께 결합
- 한국어는 보통 평균 20 subword로 구성되고 99%가 60 subword를 넘지 않기 때문에 입력 길이를 두 문장이 합쳐 128(max_len)으로 설정 해도 충분합니다
- 간혹 긴 문장이 있으므로 우선 입력 길이 128로 제한하고 학습한 후, 128보다 긴 입력들을 모아 마지막에 따로 추가 학습하는 방식을 사용

### Position Embedding

- BERT는 Transformer 모델을 착용 그 중 Self-Attention 모델을 사용
- Self-Attention은 입력의 위치에 대해 고려하지 못하므로 입력 토큰의 위치 정보가 필요(position embedding필요성)
- Position encoding은 단순하게 Token 순서대로 0, 1, 2, ...와 같이 순서대로 인코딩

## Fine Tunning의 2가지 대표적인 방법

- BERT는 위에서 소개한 3가지의 입력 임베딩(Token, Segment, Position 임베딩)을 취합하여 하나의 임베딩 값으로 생성
- 임베딩의 합에 Layer Normalization과 Dropout을 적용하여 입력으로 사용

![파인튜닝 구조](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbg5SlP%2FbtqzntBU7Uj%2FKHWiKI4zKgb8FqLzAYAusK%2Fimg.png)

### MLM(Masked Language Model)

- 입력 문장에서 임의로 Token을 마스킹(masking), 그 Token을 맞추는 방식인 MLM 학습 진행
- 문장의 빈칸 채우기 문제를 학습
- 생성 모델 계열은(예를들어 GPT) 입력의 다음 단어를 예측
- MLM은 문장 내 랜덤한 단어를 마스킹 하고 이를 예측
- 입력의 15% 단어를 [MASK] Token으로 바꿔주어 마스킹
- 이 때 80%는 [MASK]로 바꿔주지만, 나머지 10%는 다른 랜덤 단어로, 또 남은 10%는 바꾸지 않고 그대로 둠
- 이는 튜닝 시 올바른 예측을 돕도록 마스킹에 노이즈를 섞음

![학습 방법](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLMyXN%2Fbtqzl4Ql7sH%2FykzRZNWkc6rcb8ffU5Nrm1%2Fimg.png)

### NSP(Next Sentence Prediction)

- NSP는 두 문장이 주어졌을 때 두 번째 문장이 첫 번째 문장의 바로 다음에 오는 문장인지 여부를 예측하는 방식
- 두 문장 간 관련이 고려되어야 하는 NLI와 QA의 파인튜닝을 위해 두 문장이 연관이 있는지를 맞추도록 학습
- 위에서 설명한 MLM과 동시에 NSP도 적용된 문장들
- 첫 번째 문장과 두 번째 문장은 [SEP]로 구분(스페셜 토큰)
- 두 문장이 실제로 연속하는지는 50% 비율로 참인 문장과, 50%의 랜덤하게 추출된 상관 없는 문장으로 구성
- 이 학습을 통해 문맥과 순서를 학습 가능
- 아래 그림은 NSP의 입력 예시

![NSP(des)](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmRPzz%2Fbtqzps28Eyd%2F2ak5AHBLlk1jXHnOgGwyMK%2Fimg.png)

#### 코랩 환경에서 챗봇 실행 방법

1.해당 파일을 코랩에서 실행합니다.

`chatbot_think_big/thinkbig_stuff/최종_실행파일/Thinkbig_chatbot(colab).ipynb`

2.런타임 유형을 `GPU`로 세팅 후 실행하면 됩니다.

3.코랩 환경에서 챗봇 실행 동영상입니다.

[![코랩 챗봇 구동 영상](https://user-images.githubusercontent.com/111936229/206372581-a6da8be0-91fa-41d9-b28c-b7574ca9d0af.png)](https://youtu.be/HHClT36nYT8)

- 버전 문제로 로컬 환경에서는 계속 오류가 나네요
- 실행 시 에러가 날때도 있고 아닐 때도 있긴한데 이유를 모르겠습니다.
- 1epch당 약 3시간이 걸리므로 유료버전에서 돌려보는 것을 추천(아니면 중간에 끊길 확률이 높습니다.)

>**참고 자료**<br>
[ebbn] : <https://ebbnflow.tistory.com/151><br>
[NLP-kr] : <https://github.com/NLP-kr/tensorflow-ml-nlp-tf2><br>
[이수한컴퓨터연구소] : <https://www.youtube.com/watch?v=LEtLfx1dS7Q><br>
[위키독스] : <https://wikidocs.net/156998>
