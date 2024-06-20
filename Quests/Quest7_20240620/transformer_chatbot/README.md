# AIFFEL Campus Online Code Peer Review Templete
- 리뷰어 : 최호재

# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    > - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    > - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭
    >    - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    - [x] 공백과 특수문자 처리, 토크나이징, 병렬데이터 구축의 과정이 적절히 진행되었다.![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/81d0be9c-be3e-43aa-a47e-b7583d0666ee)
    - [ ] 구현한 트랜스포머 모델이 한국어 병렬 데이터 학습 시 안정적으로 수렴하였다.
        - 모델 학습 부분을 진행하지 못하셨습니다. (validation loss 의 증감을 살펴보는게 꼭 필요했던 것 같습니다.)
    - [ ] 한국어 입력문장에 맥락에 맞는 한국어로 답변을 리턴하였다.
        - 한국어 입력문장에 대한 답변을 출력하지는 못하셨습니다. ( test dataset 에서의 qustion, answer 를 비교해 보면 좋을것 같습니다.)
    
- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [x]  모델 선정 이유
        - [x] positional encoding 에 대해서 깊은 분석을 하셨습니다. ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/0b71b825-b8fb-4cc6-9100-33dcd06e30e5). `Quest7_20240620/songys_chatbot
/Transformers_practice.ipynb`
        - [x] mask, encoder, decoder 부분에 대한 분석을 주의깊게 진행하셨습니다. `Quest7_20240620/songys_chatbot
/Transformers_practice.ipynb` ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/a6011e19-24b7-4170-8c91-11e18a74e469) ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/1f03faac-7bbe-45f0-bcec-bea26b83b54c) ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/1ab2b1be-a61a-4d3b-9de8-025e625881f8)
    - [ ]  Metrics 선정 이유
    - [x]  Loss 선정 이유 ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/646dd70f-cd52-49ce-8068-010c96c9f8ef) `Quest7_20240620/songys_chatbot
/Transformers_practice.ipynb`
    
- [ ]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - 시간상의 이유로 다음 부분을 진행하지 못하셨습니다.
    - [ ]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [ ]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [ ]  각 실험을 시각화하여 비교하였나요?  
    - [ ]  모든 실험 결과가 기록되었나요?

- [x]  **4. 회고를 잘 작성했나요?**
    - [x]  배운 점
      - Transformer 모델에 대해서는 잘 이해할 수 있었다.
    - [x]  아쉬운 점
      -  2024년 6월 20일 16시 55분 기준, 예제 코드는 다 끝냈지만, 프로젝트는 시작해야하는 상황이어서 아쉽다.
    - [ ]  느낀 점
    - [X]  어려웠던 점  
      - 개별 구성요소들이 최소 함수 혹은 클래스 구현체로 되어있고, 이것들의 연계가 아직까지는 빠르게 읽히지 않음.
      - (batch_size, sequence_length, dimension_of_model) 단위(최소 2차원) 텐서 연산이 머릿속에 잘 안그려진다.

**리뷰어 의견**  
- 케라스 소스코드와 lms의 코드를 비교하신 것에서 학습 의지가 존경스러웠습니다. ![image](https://github.com/hojae-m-choi/AIFFEL_RESEARCH_STUDY/assets/98305832/9753d21d-8c09-40fa-a10a-46d043c0af89)

- 전반적으로 코드에 대해서 깊게 이해하시려는 시도들이 눈에 띄었습니다.



