---
title: Fancy Indexing 정리
description: NumPy의 팬시 인덱싱 개념 정리
created: 2025-06-23
tags: [numpy, python]
---

# (1) 기초적인 Fancy Indexing
## Fancy Indexing의 기초 개념

이젠 다른 방식의 array indexing을 배울거라네?
근데 하나의 scalar를 쓰는 대신 index들의 array를 사용할거라고 해.
이게 무슨 말일까??
```python
import numpy as np

x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
[x[3], x[7], x[2]]
# 이런 식으로 list를 만들 수 있겠지만,

ind = [3, 7, 4]
x[ind]
# 이렇게 해도 array([40, 80, 50]) 결과가 잘 나오네!

```
기존엔 array의 index 부분 안에 \[3] 이렇게 하나의 숫자. 비유하자면 scalar를 넣었는데,
**Fancy Indexing**은 <u>array의 index 부분 안에
또다시 원하는 index들의 array를 넣는 방식인거야!</u>

## index array의 shape

근데 여기서 중요한게 있어
fancy indexing은 index의 대상이 되는 array의 shape 말고,
우리가 원하는 index의 array의 shape를 반영해!
```python
ind = np.array([[3, 7],
				[4, 5]])
x[ind]
"""
array([[40, 80],
		[50, 60]])
"""
```
{지피티}
지피티에게도 이 문장을 물어봤었는데, 이 부분이 왜 중요한가싶더니
x의 shape에는 영향을 주지 않으면서 index 배열의 shape로만 뽑아오는거야
이게 일반 슬라이싱의 **원본 배열에서 선택된 범위를 가져오는거랑 차이가 있는거래.** [[NumPy 2 The Basics of NumPy Arrays#Slicing]]

## Fancy Indexing + Broadcasting

```python
x = np.arange(9).reshape((3, 3))
rows = np.array([[0], 
				[1]])
cols = np.array([[1, 2]])
x[rows, cols]
"""
[rows, cols]가
[[0, 1], [0, 2],
[1, 1], [1, 2]]가 되고,
x[rows, cols]가 broadcasting 되어서
array([[1, 2],
		[4, 5]])
"""
```
indices array 자체가 broadcasting 되는 경우도 볼 수 있지
***broadcasting 태그 미완성***

# (2) Combined indexing

2차원 배열이면 이런 기묘한 일도 일어나
```python
X = np.arange(12).reshape((3, 4))
X[2, [2, 0, 1]]
# array([10, 8, 9])
X[1:, [2, 0, 1]]
# 이러면 두줄짜리가 나오겠네
```

```python
mask = np.array([1, 0, 1, 0], dtype=bool)
row = np.array([0, 1, 2])
X[row[:, np.newaxis], mask]
```
이런걸 indexing by masking이라고 한대
*진짜 Python이 유연한건 원탑이네*

### indexing by masking 얘깃거리

난 이게 정말 이해가 안 됐는데, 
일단 row랑 mask를 통해 3 x 4 boolean mask shape가 나온건 맞아
근데 실제 인덱싱 과정에서 True인 열만 선택된다는게....
*mask에 아무 숫자가 안 적혀있는데 대체 어떻게 열이 \[0, 2]인걸 아는거지?*

{지피티}
Boolean 배열은 인덱스로 쓸 때, **자동으로 True인 위치의 인덱스 번호로 바뀐대**
```python
np.where(mask)
# array([0, 2], )
```
결국은 이렇게 변환이 되더라...

#### np.where(condition)

참고로 얘는 True인 위치(인덱스)를 반환하는 함수래
np.where(condition, value_if_true, value_if_false)는 
condition 만족하면 2번째 값, 만족 안 하면 3번째 값 나오는 애야


## 통계와 연결한 예시 (슬슬 통계 나오네)

XY좌표로 된 점 100개를 특정 공분산 상에서 찍어보는 연습을 해보자고

### rand.multivariate_normal(mean, cov, sample)

```python
mean = [0, 0]
cov = [[1, 2],
		[2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
```
이러면 2차원 점 100개를 저 공분산 행렬 cov를 통해 랜덤으로 찍을 수 있어
여기서의 공분산 행렬이란
\[Var(차원1), CoV]
\[Cov, Var(차원2)]

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.scatter(X[:, 0], X[:, 1]);
```
현재 X의 shape는 2차원짜리 점 100개를 찍었으니 (100, 2)
점의 개수 N이 행의 개수가 되고,
차원 D가 열의 개수가 되는건 관습이래.

아직 우리가 matplotlib을 배운 상태는 아니지만...
X\[:, 0]을 하면 index 0열. 즉 x좌표가 다 나오고,
X\[:, 1]을 하면 index 1열. 즉 y좌표가 다 나온다!
scatter는 그걸 다 찍겠단 소리곘지

![[Pasted image 20250530053409.png]]
이제 그림을 그렸으니 여기에서 점을 뽑자는거야
```python
indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]
```
**와씨 행 몇 개인지 구하는걸 X.shape\[0]로 하네?**
replace=false는 대충 중복 없이 하란 소리 같고.

```python
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='none', edgecolor='r', s=50);
```
![[Pasted image 20250530055834.png]]
위에랑 똑같은 문법으로 random으로 고른 점 20개를 찍을건데,
alpha는 투명도, facecolor는 바탕색이랑 같이 할건지 말건지, 
edgecolor는 모서리 색깔, s는 사이즈를 의미한대! 와! 예습!

# (3) Fancy Indexing + 값 바꾸기

Fancy Indexing은 여러 값을 불러올 뿐만 아니라, 여러 값을 바꿀 수도 있는데...
```python
x = np.arange(5)
i = np.array([2, 1, 4])
x[i] = 99
# [0, 99, 99, 3, 99]
```
사실 이러면 i의 2, 1, 4 순서가 의미가 있겠냐싶긴 하다...

```python
x[i] -= 10
# [0, 89, 89, 3, 89]
```
물론 이렇게 값을 빼는 것도 가능하겠지

### assignment-type operator

책에서 이런걸 assignment-type operator라고 하던데 이게 뭘까?
우리가 대학에서 엄청 많이 배웠던 +=처럼 축약한 연산자인데 멋지게 이름 붙인거야
뒤에 =이 붙는 += \*= 같은 모든 애들을 얘기하지!
고급스럽게 얘기하면 <u>기존 값으로 다시 할당하는 축약 연산자</u>인거야야

## index array에 같은 숫자가 있다면?

```python
x = np.zeros(5)
x[[0, 0]] = [1, 2]
# [2. 0. 0. 0. 0.]
```
이런 경우에는 x\[0] = 1랑 x\[0] = 2가 이어져서 
x\[0]이 마지막에 나온 2란 값을 가지게 됐어
이건 딱히 이상해보이진 않는데...

```python
i = [2, 2, 2]
x[i] += 1
# array([2., 0., 1., 0., 0.])
```
뭐냐 이거 왜 index 2에서 3이 안나오냐?
왜 좀 전의 x\[\[2, 1, 4]] -= 10은 잘 작동하고,
x\[\[2, 2, 2]] += 1은 왜 작동하지 않냐??
\[\[2, 2, 2]]가 \[2]로 먼저 변환되기라도 하나??

x\[i] += 1가 x\[i] = x\[i] + 1이잖아?
사실 이 **assignment-type operator라는 놈은** 단순히 값을 더하고 끝내는게 아니라
**복사해서 지정된 연산을 하고 다시 할당해주는 애였대**
알고보니 이 코딩에선 x\[2]를 세 번 복사해서, 각 x\[2]에 +1을 해주고, 똑같은 x\[2] + 1 값을 3번이나 원래 x\[2] 위치에 대입하는 일이 일어나는거지
즉, [[단어장#^augmentation]]이 아닌, [[단어장#^assignment]]가 일어나는거지

### np.add.at(array, indices, number)

만약 진짜 여러번 더하고 싶다면 이걸 이용하는게 나아
```python
x = np.zeros(5)
i = [2, 2, 2]
np.add.at(x, i, 1)
# array([0., 0., 3., 0., 0.])
```
at() 메소드는 in-place application을 수행해.
즉, 위에처럼 복사하거나 그런게 아니라 그 자리에서 주어진 연산자의 계산을 수행해!

### reduceat(array, indices, axis)

비슷하게 reduceat()이라는 애도 있던데,
누적 연산을 reduction이라고 하거든?
얘는 여러 인덱스에서 누적 연산을 구간별로 나눠서 적용하는 함수래
```python
import numpy as np
x = np.arange(7)
np.add.reduceat(x, [0, 3, 5])
# array([0 + 1 + 2, 3 + 4, 5 + 6])
```
0부터 3 전까지 index 합친거
3부터 5 전까지 index 합친거
그 뒤로 다 합친거인거지

#### 호기심 뇌절

runebook이라는 데에서 좀 더 검색해보고 복사해왔는데... 
```python
>>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
array([ 6, 10, 14, 18])
```
이렇게 뒤에 \[::2]를 붙여서 0~4, 1~5, 2~6, 3~7로 2개씩 끊을 수도 있는 것 같고...
만약 \[::2]가 안나왔으면, 어떻게 될지는 다음 예시 보면 알 수 있어.

```python
>>> x = np.linspace(0, 15, 16).reshape(4,4)
>>> np.add.reduceat(x, [0, 3, 1, 2, 0]) # axis는 기본으로 0. 행 기준이다.
array([[12.,  15.,  18.,  21.],
       [12.,  13.,  14.,  15.],
       [ 4.,   5.,   6.,   7.],
       [ 8.,   9.,  10.,  11.],
       [24.,  28.,  32.,  36.]])
```
2차원 배열이면 이런 일이 벌어지는데...
1. 0번째 행부터 3번째 행 전까지 다 더한거
2. 3번째 행부터 1번째 행까지는 말이 안 되니까 3번째 행만 출력
3. 1번째 행부터 2번째 행 전까지 다 더한거. 즉 1번째 행만 출력
4. 2번째 행부터 0번째 행까지는 말이 안 되니까 2번째 행만 출력
5. 0번째 행이 맨 끝이니까 0번째 행부터 마지막까지 다 더한거
이런 식으로 작동한단걸 알 수 있지.



# (4) Example: Binning Data

Data binning이 뭔가 하고 봤더니... 사소한 관측 오차의 영향을 줄이는 데 사용되는 데이터 전처리 기술이래
{지피티}
실제로 데이터를 분석할 때 연속된 수치를 구간. 범주. bin으로 나누는거래!
연속을 이산적으로 나누는 느낌이려나??

이건 내가 뭐하는건지 모르겠으니... 그냥 예시만 따라 치고 모르는 함수만 적어두자

#### np.random.randn(m, n)
평균 0, 표준편차 1의 표준정규분포 난수를 만드는데, 그걸 m행 n열로 만들어라!
```python
np.random.seed(42)
x = np.random.randn(100)
```

#### np.zeros_like(array)
안에 넣은 array랑 같은 크기의 0으로 가득 찬 array를 만든다!
```python
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
```
복습 함수: \[np.linspace]

#### np.searchsorted(array_A, array_B)
array_A를 기준으로 array_B의 값들이 어디에 위치할지를 알려준다!
A가 \[1, 2, 3], B가 \[1.5, 3]이면 1.5면 1과 2 사이에 위치. 3은 index 2에 그대로 있으니...
이 메소드를 쓰면 index로 만들어진 array \[1, 2]가 될거야

1.5가 먼저 1과 2 사이에 들어갔다고 생각 안하고!
그냥 1, 2, 3 있는데 1.5하고 3이 어디에 낄 수 있을까 찾아주는거야.
```python
i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)
```
복습 함수: \[np.add.at]
자 이제 무슨 함수인지 알겠지? 
searchsorted 메소드를 구간을 찾아주고,
zeros_like 메소드로 미리 만든 0짜리 array에
add.at 메소드로 그 구간 부분에 1씩 더해줘서 데이터를 구간 별로 나누는 binning을 해준다!

```python
plt.hist(bins, counts, histtype='step');
```
![[Pasted image 20250604043251.png]]

근데 사실? np.histogram(x, bins)라는 방식으로 그려도 된대!
그치만 이게 %timeit을 해보면 알겠지만, 우리가 한 방식이 더 빨라!
그럼 이거 np.histogram은 왜 쓰는거야?

잘은 모르겠지만 NumPy 쪽의 알고리즘이 좀 더 유연하대
그래서 데이터 수가 많아지면 NumPy 쪽이 성능이 더 좋다는데,
이건 마치 O(n) 같은거 해봤을 때 초반만 NumPy가 밀리는게 아닐까 싶어!
```python
%timeit counts, edges = np.histogram(x, bins)
%timeit np.add.at(counts, np.searchsorted(bins, x), 1)
```
이렇게 하면 NumPy 쪽이 2배 더 빠르다곤 하는데..

근데 timeit을 할 때 윗 줄은 왜 저렇게 적은거지??
아냐.. 다시 생각해보니 별게 아냐...
그냥 np.histogram(x, bins)라는 애가 한번에 두개의 변수를 만든거야
아마 우리가 만든 counts도 만들면서, edges라는 히스토그램을 만든거지!
그래서 저 코드 자체가 한줄인 것 같아!
{지피티}
질문하는 동안 내가 스스로 깨닫게 된 질문이지만,
여담으로 지피티가 저 %timeit을 할 때 변수가 저장이 안 된대
나중에 실제로 결과를 저장할거면 변수 지정을 해줘야된다네!
그리고 위의 내가 궁금했던 상황을 %timeit은 다중 할당도 지원한다고 표현한대

# 지피티 연습문제

```python
import numpy as np

# Step 1. 0부터 99까지의 정수로 이루어진 배열 x를 생성하라.
x = np.arange(100)

# Step 2. 난수 시드를 0으로 고정하고, x에서 중복 없이 10개의 원소를 무작위로 뽑아 indices로 저장하라.
np.random.seed(0)
indices = np.random.choice(100, 10, replace=False)
print(indices)

# Step 3. 이 indices를 활용해, x에서 해당 위치의 값들을 999로 바꾸어라.
x[indices] = 999
print(x)

# Step 4. 바뀐 x 배열에서, 999가 저장된 위치의 인덱스들만 뽑아 result 배열로 만들어라.
# Step 5. result 배열을 오름차순으로 정렬해서 출력하라.
boolean = (x == 999)
result = np.where(boolean)[0]
print(result)
```
Step 4의 의도가 이상해보였는데, indices를 까먹었다 생각하고 다시 구하라는 의미지!
Step 3가 Fancy Indexing 해본거고, 그걸 다시 뽑아내는게 문제였네
If문 했으면 편했겠지만, 그런거 아직 안 배웠다 치고 해봤어.
result = np.where(x = 999)\[0] 하면 좀 더 깔끔했겠다
여담으로 저 boolean은 index 0이 array고, 그 다음엔 type이 나오더라고.

