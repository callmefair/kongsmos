---
title: Pandas 1
description: 간단한 설명
draft: false
tags: [공부, ai, 수학]
---
이제부터 우리가 배울건 NumPy를 기초로 해서 만들어진 **DataFrame**
이 DataFrame이란건 서로 다른 data들이 행렬로 <u>label화 되어있는 multi-dimensional array</u>

우리가 지금까지 배웠던 ndarray가 배우는데 문제는 없었는데... 이게 그리 유연하진 않대
data에 label을 붙인다던가, 누락된 data를 찾는다던가 그런 일에 적합하지 않다네
그리고 DataFrame이 ndarray에서 broadcasting 안되는 것도 잘 한대

이런건 앞으로 DataFrame 배우면 얘는 뭐가 그리 잘났는지 알 수 있겠지
데이터 분석 배우는데 이왕이면 상위호환인거 배우면 더 좋고.

아마 Series란거 먼저 배우고 DataFrame 배울건가봐. 가보자고~
```python
import numpy as np
import pandas as pd
```

# (1) 1차원의 Series
## 1-1. 기초적인 Series
### pd.==Series==(list)

자 이제 label을 하나씩 붙여보는거야
Series는 indexed. 즉 label 이름이 붙은 data인데 1차원인거!
```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
```
![[Pasted image 20250629182506.png]]
ndarray랑은 다르게, 이렇게 series에는 value 뿐만 아니라 index도 들어있단걸 알 수 있어

### series.==values== 그리고 series.==index==

```python
data.values
# array([ 0.25,  0.5 ,  0.75,  1.  ])
```
일단 value 자체는 그냥 ndarray 느낌

```python
data.index
# RangeIndex(start=0, stop=4, step=1)
```
근데 labeled라길래 다 string한 이름 붙어있거나 그런줄 알았는데,
꽤나 노골적으로 index네?
*근데 왜 저렇게 번거롭게 step식으로 붙여놨을까?*

```python
data[1:3]
"""
1    0.50
2    0.75
dtype: float64
"""
```
accessing이나 slicing도 label까지 잘 가져오는 모습이다!

### pd.Series(..., ==index\[]==)

근데 이거 그냥 NumPy array로도 하고있던거 아냐?
그건 맞지. 근데 말했다시피 Pandas는 label을 달 수 있다고 했지?
저렇게 위처럼 label이 index랑 다른게 없다면 Pandas를 배울 필요가 없겠지?
```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
					index=['a', 'b', 'c', 'd'])
data
```
![[Pasted image 20250629183703.png]]
이런게 가능해. 그리고 물론...
```python
data['b']
# 0.5
```
accessing도 잘 되는 모습. slicing은 좀 나중에 보자.

이를 두고, **NumPy Array**는 <u>implicitly하게 정의된</u> integer <u>index</u>를 가지고 있고,
**Pandas Series**는 <u>explicitly하게 정의된 index</u>를 가지고 있다고도 하네

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
```
가장 처음에는 index가 0, 1, 2, 3으로 됐는데,
label을 숫자로 해도 될 뿐더러, 당연히 숫자가 연속적일 필요도 없지
```python
data.index
# Int64Index([2, 5, 3, 7], dtype='int64')
```
*아까 data.index할 때는 step이 붙던데,
index=\[2, 4, 6, 8]로 해도 step 같은거 안나오더라고?
index 안 정할 때만 RangeIndex type 나오나보네*

## 1-2. dictionary로 만들어진 Series
### pd.Series(==dictionary==)

이것도 그냥 dictionary 같아보이겠지만, 
C struct에 바로 내리꽂아버릴 수 있다는 장점이 있지 않을까? ~~(나중에 NumPy쪽 링크 필요)~~
```python
population_dict = {'수도권': 26071579,
	                '동남권': 7566068,
	                '충청권': 5558682,
	                '대경권': 4878931,
	                '전남권': 3183996,
		            '전북권':1731309,
		            '강원권':1511341,
		            '제주권':667242}
population = pd.Series(population_dict)
population
```
![[Pasted image 20250629185442.png]]
pd.Series() 안에 dictionary를 넣으면 자동으로 Series를 만들어주네!

이 방식 쓰면 원래 dictionary에서 쓰는 accessing도 가능하고,
놀랍게도 기존 dictionary 때는 말도 안 되던 slicing이 여기선 암묵적으로 index가 있어서 가능하지!
```python
population['제주권']
# 667242
population['충청권':'강원권']
```
![[Pasted image 20250629185827.png]]

## 1-3. Series 만들기 추가 잡기술
### pd.Series(==value==, ...)

```python
pd.Series([2, 4, 6])
pd.Series([2, 4, 6], index=[100, 200, 300])
```
이런 식으로는 앞에서 배웠던거지만...

```python
pd.Series(5, index=[100, 200, 300])
```
![[Pasted image 20250629190302.png]]
이러면 모든 값을 5로 넣을 수 있어!

### pd.Series(==dictionary==, ==index=[]==)

```python
pd.Series({2:'a', 1:'b', 3:'c'})
```
이런 식으로 Series 안에 dictionary 넣는 것까진 좋았는데...

```python
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
```
이렇게 뒤에 index까지 넣는다면 dictionary에서 원하는 index만 뽑아먹을 수 있겠지

# (2) 2차원의 DataFrame
## 2-1. 기초적인 DataFrame

DataFrame은 이제 indices가 NumPy와 다르게 flexible한데 
Series하고는 다르게 two-dimensional인거!
<u>그러니 행과 열 모두 label이 붙겠지!</u>
필자는 이를 aligned된. 정렬된 Series object라고 하네.
Series가 옆으로 쭉 이어져있다는 소리겠지

### pd.==DataFrame==(==columnName: Series==)

```python
population_dict = {'수도권': 26071579,
	                '동남권': 7566068,
	                '충청권': 5558682,
	                '대경권': 4878931,
	                '전남권': 3183996,
		            '전북권':1731309,
		            '강원권':1511341,
		            '제주권':667242}
population = pd.Series(population_dict)
```
우리가 처음에 이런 series를 만들었었고,
```python
area_dict = {'수도권': 605, '동남권': 12344, '충청권': 16572,
			'대경권': 19910, '전남권': 12596, '전북권': 8055,
	        '강원권': 16613, '제주권': 1848}
area = pd.Series(area_dict)
```
이제 이런 series도 만든다고 하면...
```python
region = pd.DataFrame({'population': population,
						'area': area})
region
```
이렇게 우리가 첫 Dataframe을 만들었고, 그 모습은...
![[Pasted image 20250629192037.png]]
콘솔에서는 보기 힘든 이쁜 색깔의 결과물이 나온다. 이게뭐람

### DataFrame.==index==

```python
region.index
# Index(['수도권', '동남권', '충청권', '대경권', '전남권', '전북권', '강원권', '제주권'], dtype='object')
```
아무래도 **index**에는 <u>행의 이름</u>이 나오는거 같고...

### DataFrame.==columns==

```python
region.columns
# Index(['population', 'area'], dtype='object')
```
**columns**에는 <u>열의 이름</u>이 나오는거 같다

```python
region['area']
```
기본적으로 DataFrame은 column으로 accessing할 수 있어!

```python
region[0]
# KeyError: 0
```
아쉽게도 이러면 오류가 떠!
사실 column 이름하고 index 이름이 같으면 문제가 생길 수 있으니 column만 하게 했을 수 있지.
근데 생각해보면 NumPy에선 data\[0] 같은거 하면 첫 **행**을 내보내잖아?
근데 얘는 이런 \[] 문법으로는 accessing이 열 밖에 안돼!
아무튼 이런 특성 때문에 Dataframe을 array 말고 일반화된 dictionary로 보라고도 해.

## 2-2. DataFrame 만들기

### pd.DataFrame(==Series==, ...)

그냥 원래 있던 series를 dataframe으로 만들 수도 있겠지
```python
pd.DataFrame(population, columns=['population'])
```
![[Pasted image 20250630153146.png]]
1차원 억지로 2차원 만들기!

### pd.DataFrame(==dictionary==)

혹은 dictionary의 list를 집어넣어도 돼
난 처음에 dictionary 집어넣는줄 알았는데, 이거 꽤나 헷갈리더라
```python
data = [{'a': i, 'b': 2 * i}
		for i in range(3)]
print(data)
# [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}]
# dictionary가 아니라 dictionary의 list다!
pd.DataFrame(data)
```
![[Pasted image 20250613053909.png]]
어떻게 또 자동으로 index를 판단해서 DataFrame이 완성된 모습이다!

근데 list에서 지정을 안 해줘서 <u>DataFrame에 뻥 뚫린 부분이 있다면?</u>
```python
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
```
보면, 0 index 행의 c 열이 없고, 1 index 행의 a 열이 없으니까...
![[Pasted image 20250615115604.png]]
<u>이렇게 NaN 값으로 채워지지!</u>

### pd.DataFrame(==NumPy array==, ...)

import numpy as np가 되어있다면...
~~사실 지금까지 numpy 불러오기 없이 pandas가 작동하는지는 잘 몰라. 한번도 빼놓지 않고 같이 있더라고.~~
```python
pd.DataFrame(np.random.rand(3, 2),
             columns=['col', 'umns'],
             index=['in', 'de', 'x'])
```
![[Pasted image 20250615120052.png]]
이렇게 랜덤으로 DataFrame의 값을 만들어낼 수도 있겠네!

```python
pd.DataFrame(columns=['col', 'umns'],
             index=['in', 'de', 'x'])
```
![[Pasted image 20250630154149.png]]
여담으로 값 지정 없이 label만 넣으니까, 모든 값이 NaN이 나오고...

column, index의 label만 바꾸고 싶으면 어떻게 할까 했는데,
```python
DF = pd.DataFrame(np.random.rand(3, 2),
             columns=['col', 'umns'],
             index=['in', 'de', 'x'])
print(DF)
DF.columns = ['칼', '럼']
DF.index = ['인', '덱', '스']
DF
```
![[Pasted image 20250615120649.png]]
역시 파이썬답게 그냥 생각대로 하니까 바뀐다!

우리가 배웠던 **structured array도** 그냥 DataFrame으로 넣을 수 있지
```python
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
# 이러면 A는 [(0, 0.0), (0, 0.0), (0, 0.0)]. 대충 뒤에 dtype 설명 나올거고.
pd.DataFrame(A)
```
![[Pasted image 20250630154710.png]]
이 경우에는 굳이 columns label을 안 정해도 됐다!
~~나중에 structured 링크 하자~~

# (3) Index 그잡채
## 1. Index를 지정해보자
### pd.==Index==()

index 자체를 object로 명시해보는거야
```python
ind = pd.Index([2, 3, 5, 7, 11])
ind
# Int64Index([2, 3, 5, 7, 11], dtype='int64')
```
index 자체는 array처럼 작동한대
accessing, slicing 같은건 다 먹히겠지.
```python
ind[1]
# 3
ind[::2]
# Int64Index([2, 5, 11], dtype='int64')
```

그리고 작동하는 attribute도 좀 NumPy스럽지
```python
print(ind.size, ind.shape, ind.ndim, ind.dtype)
# 5 (5,) 1 int64
```

그치만? index를 바꾸려고 하면, tuple 마냥 못 바꾸게 한다!
```python
ind[1] = 0
# TypeError
```

index 이름이 숫자가 아니여도 잘 작동하겠지??
```python
ind = pd.Index(['떡볶이', '순대', '튀김', '어묵', '쿨피스'])
ind
# Index(['떡볶이', '순대', '튀김', '어묵', '쿨피스'], dtype='object')
```
뭔가 위의 숫자 index 때랑은 결과 생긴게 달라보이지만.. 아무튼 pandas라 되는 모습!

같은 숫자 있어도 되더라?
```python
ind = pd.Index([1, 1, 2, 3, 5, 8])
ind
# Int64Index([1, 1, 2, 3, 5, 8], dtype='int64')
```
## 2. Index 주제에 논리연산이 된다

왜 있는지 모르겠지만 이 Index들끼리 and, or 같은걸 할 수가 있어!
무슨 순서 있는 집합?처럼 만들어진 애들인가봐. 그래서 ordered set이라는 표현도 나오네
뭐 사실 boolean 같다고 표현할 수도 있겠네
```python
indA = pd.Index([2, 3, 5, 7, 11])
indB = pd.Index([1, 1, 2, 3, 5, 8])
indA & indB
# 얘는 intersection. Int64Index([3, 5, 7], dtype='int64')
indA | indB
# 얘는 union
indA ^ indB
# 얘는 서로의 차집합의 합집합? 아무튼 그런거
```

