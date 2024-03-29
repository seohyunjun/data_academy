# KDATA Data Academy 2022 :1st_place_medal:

## 센서 신호분석을 통한 이상분류 모델 개발([기계시설물 고장 예지 센서 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=238))
<br><br/>

## :books:Data info
- 정상데이터: 1,098,327
- 고장데이터 
  - 베어링 불량:    367,929
  - 회전체 불평형:  223,502
  - 축정렬 불량:    330,651
  - 벨트 느슨함:    312,763
<br><br/>

## :pushpin:Classification Info
> ### 1 휴리스틱 기준으로 분류 가능
     3    정상 & 회전체불평형(2.2) L-EF-04 (구분이 되는 듯 안되는 유형)
     17   정상 & 회전체불평형(5.5) R-CAHU-01R
     18   정상 & 축정렬불량(5.5) R-SF-01   
     26   정상 & 벨트느슨함(11) R-CHAU-02R
       - Constant K 설정
       - K 기준 데이터의 99%가 위에 있으면 불량, 데이터의 99%가 아래에 있으면 정상
<img width="100%" height="60%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/vibration_peak_plot3/3_vibration_2.2_L-EF-04_%ED%9A%8C%EC%A0%84%EC%B2%B4%EB%B6%88%ED%8F%89%ED%98%95.jpg?raw=true"/>

<img width="100%" height="30%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/model_3_nugabado_gubyul.gif?raw=true"/>
<br><br/>

> ### 2A CR(Coverage Ratio) 이용하여 분류 가능
     2    정상 & 축정렬불량(2.2) L-DSF-01
     4    정상 & 베어링불량(2.2) L-SF-04
     8    정상 & 벨트느슨함(2.2) R-SF-03
     10   정상 & 베어링불량(3.7) L-EF-02
     16   정상 & 베어링불량(5.5) L-SF-02
     19   정상 & 회전체불평형(7.5) L-PAC-01
     22   정상 & 축정렬불량(7.5) R-PAC-01S
     23   정상 & 축정렬불량 & 회전체불평형(11) L-CAF-01R
     25   정상 & 벨트느슨함(11) R-CAHU-01R
     31   정상 & 베어링불량(18.5) R-CAHU-02S
     32   정상 & 축정렬불량(22) L-CAHU-01S
     37   정상 & 회전체불평형(55) L-PAHU-03S
     def. CR각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 밖으로 나가있는지 비율로 계산
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/vibration_peak_plot3/2_vibration_2.2_L-DSF-01_%EC%B6%95%EC%A0%95%EB%A0%AC%EB%B6%88%EB%9F%89.jpg?raw=true"/>
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/model_2_Coverage_Ratio.gif?raw=true"/>
<br><br/>

> ### 2B iCR 사용해야하는 경우
     7    정상 & 벨트느슨함(2.2) R-EF-05
     29   정상 & 벨트느슨함(15) R-CAHU-03S
     30   정상 & 벨트느슨함(18.5) R-CAHU-01S
     34   정상 & 벨트느슨함(22) R-CAHU-02S
     38   정상 & 벨트느슨함(55) R-PAHU-04S 
    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 안으로 들어가있는지 비율로 계산
    CR = (200 - in point) / 200: 1에 가까울수록 정상에 가까움
			    0에 가까울수록 비정상에 가까움
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/vibration_peak_plot3/7_vibration_2.2_R-EF-05_%EB%B2%A8%ED%8A%B8%EB%8A%90%EC%8A%A8%ED%95%A8.jpg?raw=true"/>
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/model_7_iCR_%EC%82%AC%EC%9A%A9.gif?raw=true"/>
<br><br/>

> ### 3 완전 알 수 없는 경우
    12 정상 & 회전체불평형(3.7) L-PAC-01
    21 정상 & 벨트느슨함(7.5) R-CAHU-03R
    23 정상 & 축정렬불량 &  회전체불평형(11) L-CAHU-01R
    27 정상 & 회전체불평형(15) L-CAHU-01S
    30 정상 & 벨트느슨함(18.5) R-CAHU-01S
    33 정상 & 회전체불평형(22) L-CAHU-02S
    35 정상 & 축정렬불량(30) R-PAHU-03S
    36 정상 & 축정렬불량(37) L-PAHU-02S
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/vibration_peak_plot3/12_vibration_3.7_L-PAC-01_%ED%9A%8C%EC%A0%84%EC%B2%B4%EB%B6%88%ED%8F%89%ED%98%95.jpg?raw=true"/>

<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/model_12_nodab.gif?raw=true"/>
<br></br>

> ### 4 정상만 있는 경우
    1,5,6,9,11,13,14,15,20,24,28
  
<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/plot/vibration_peak_plot3/1_vibration_2.2_L-DEF-01_%EC%A0%95%EC%83%81.jpg?raw=true"/>

- 육안으로 구분할 수 없는 데이터 정상 신호를 AE로 만들어 비교 map와 큰 차이 날 경우 이상치로 분류
- plot disp -> abs(min) constant line
<br></br>


> LSTM-AE

<img width="100%" src="https://github.com/seohyunjun/data_academy/blob/main/lstm_autoencoder_disp.png?raw=true"/>
<br><br/>

## Reference Material
- [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148)
- [Jittering](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/relationships?ex=4https://blog.naver.com/statstorm/222410116082)
- [sampling 방법](https://blog.naver.com/pmw9440/222414568243)


<!-- 
## Power Point
https://docs.google.com/presentation/d/1Yr0i_iTTahdRIN6ClQu0KhNzEUtT0Er-BmW5j29_XnU/edit#slide=id.ge69f820f30_0_33 -->

<!-- ~11일까지 
https://docs.google.com/presentation/d/1Yr0i_iTTahdRIN6ClQu0KhNzEUtT0Er-BmW5j29_XnU/edit#slide=id.ge7f8985915_0_83 -->



