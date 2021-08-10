## 데이터 아카데미 교육 프로젝트 

Vibration_ANALYSIS 
* PEAK VALUE COUNT
## 1. 누가봐도 눈으로 구별이 되는경우
     3    정상 & 회전체불평형(2.2) L-EF-04 (구분이 되는 듯 안되는 유형)
     17   정상 & 회전체불평형(5.5) R-CAHU-01R
     18   정상 & 축정렬불량(5.5) R-SF-01   
     26   정상 & 벨트느슨함(11) R-CHAU-02R
       - Constant K 설정
     K 설정하여 99% 이상 데이터가 위에 있으면 불량, 99% 이하 데이터가 아래에 있으면 정상

## 2A. Coverage Ratio를 사용해야하는 경우
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
    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 밖으로 나가있는지 비율로 계산
    
    CR = (200 - Out point) / 200: 1에 가까울수록 정상에 가까움
			      0에 가까울수록 비정상에 가까움

## 2B. iCR 사용해야하는 경우
<<<<<<< HEAD:README.txt
>     7,29,30,34,38
>    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 안으로 들어가있는지 비율로 계산
>    CR = (200 - in point) / 200: 1에 가까울수록 정상에 가까움
>			    0에 가까울수록 비정상에 가까움

## 2C. CR 공통 idea
>    minus 진동값은 절대값을 이용하여 평균 계산 
=======
     7    정상 & 벨트느슨함(2.2) R-EF-05
     29   정상 & 벨트느슨함(15) R-CAHU-03S
     30   정상 & 벨트느슨함(18.5) R-CAHU-01S
     34   정상 & 벨트느슨함(22) R-CAHU-02S
     38   정상 & 벨트느슨함(55) R-PAHU-04S 
    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 안으로 들어가있는지 비율로 계산
    CR = (200 - in point) / 200: 1에 가까울수록 정상에 가까움
			    0에 가까울수록 비정상에 가까움
>>>>>>> ad4877899382e25a29daa1c8ed341c754539c1b6:README.md

## 3. 완전 알 수 없는 경우
    12 정상 & 회전체불평형(3.7) L-PAC-01
    21 정상 & 벨트느슨함(7.5) R-CAHU-03R
    23 정상 & 축정렬불량 &  회전체불평형(11) L-CAHU-01R
    27 정상 & 회전체불평형(15) L-CAHU-01S
    30 정상 & 벨트느슨함(18.5) R-CAHU-01S
    33 정상 & 회전체불평형(22) L-CAHU-02S
    35 정상 & 축정렬불량(30) R-PAHU-03S
    36 정상 & 축정렬불량(37) L-PAHU-02S

   추후 연구과제

## 4. 정상만 있는 경우
    1,5,6,9,11,13,14,15,20,24,28

   추후연구과제




[0807]
## 
Plot Min 부분을 절댓값을 취해 상수와 같은 하나의 선으로 표현 

## 
Jittering
https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/relationships?ex=4
https://blog.naver.com/statstorm/222410116082


##
완전히 알 수 없는 경우

##
sampling 방법
https://blog.naver.com/pmw9440/222414568243


## Power Point
https://docs.google.com/presentation/d/1Yr0i_iTTahdRIN6ClQu0KhNzEUtT0Er-BmW5j29_XnU/edit#slide=id.ge69f820f30_0_33

~11일까지 
https://docs.google.com/presentation/d/1Yr0i_iTTahdRIN6ClQu0KhNzEUtT0Er-BmW5j29_XnU/edit#slide=id.ge7f8985915_0_83
