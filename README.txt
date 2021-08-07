## 데이터 아카데미 교육 프로젝트 

Vibration_ANALYSIS 
* PEAK VALUE COUNT
## 1. 누가봐도 눈으로 구별이 되는경우
     3,,17,18,26  - Constant K 설정
     K 설정하여 99% 이상 데이터가 위에 있으면 불량, 99% 이하 데이터가 아래에 있으면 정상

## 2A. Coverage Ratio를 사용해야하는 경우
     2,4,8,10,16,19,22,23회전체,25,31,32,37
    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 밖으로 나가있는지 비율로 계산
    
    CR = (200 - Out point) / 200: 1에 가까울수록 정상에 가까움
			      0에 가까울수록 비정상에 가까움

## 2B. iCR 사용해야하는 경우
>     7,29,30,34,38
>    각 포인트 별로 (엑셀파일 각 값) 평균값을 산출하여 하나의 선을 만들고, 그 데이터에서 얼마나 안으로 들어가있는지 비율로 계산
>    CR = (200 - in point) / 200: 1에 가까울수록 정상에 가까움
>			    0에 가까울수록 비정상에 가까움

## 2C. CR 공통 idea
>    minus 진동값은 절대값을 이용하여 평균 계산 

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

