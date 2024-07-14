# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이정훈
- 리뷰어 : 윤소정


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 1. multiface detection을 위한 widerface 데이터셋의 전처리가 적절히 진행되었다.	tfrecord 생성, augmentation, prior box 생성 등의 과정이 정상적으로 진행되었다.
      <br>![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/62fd59ec-fb5a-457e-98e0-a1040e5a019e)<br>

        - 2. SSD 모델이 안정적으로 학습되어 multiface detection이 가능해졌다.	inference를 통해 정확한 위치의 face bounding box를 detect한 결과이미지가 제출되었다. <br> ![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/dc46ac4f-8160-4776-a03c-56360c95987a)<br>

        - 3. 이미지 속 다수의 얼굴에 스티커가 적용되었다.	이미지 속 다수의 얼굴의 적절한 위치에 스티커가 적용된 결과이미지가 제출되었다.<br>
        ![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/58323335-01bc-4d3d-97ec-8c96a035ee49)<br>

    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        ![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/a85b4516-0053-41e0-b112-15f11b2394e1)
        - 주석을 잘 달아주셨습니다.
          
- []  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - 디버깅이 있었지만 해결한 기록은 없었습니다.
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - ![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/aecd5215-de9e-436b-a4c2-b575f395b555) <br>
        

        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - ![image](https://github.com/coronarita1991/AIFFEL_RESEARCH_STUDY/assets/104029654/c797f1bd-0751-425e-bd23-616a7645cd70) <br> 스티커 붙이는 함수를 따로 만들어 주셨습니다.



_________________________________________________________________
### 리뷰어 회고:
저는 얼굴 탐지기가 제대로 작동하지 않아서 모델을 만드는데 시간을 너무 소비했는데 다음부터는 정훈님처럼 되는 것 부터 먼저해야겠습니다! 수고하셨습니다!

