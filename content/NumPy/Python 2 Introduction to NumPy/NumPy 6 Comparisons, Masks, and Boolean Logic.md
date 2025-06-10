Boolean mask? NumPy array의 value를 조사하고 조작하기 위한거라네?
Boolean인거 보니까 yes or no로 나올거 같네

뒤에서 나올 pandas import를 잠깐 쓰자면...
이걸 보아하니 pandas는 데이터 가져오는 용도 같네
```python
import numpy as np
import pandas as pd

rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0
inches.shape
```

seaborn도 결국 나왔어. 애는 아무래도 그래프 그리는거랑 관련이 있나봐
```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(inches, 40);
```
![[Pasted image 20250522211752.png]]
근데 이런 비 몇 인치 왔는지는 사실 좋은 통계는 아니지. 딱 봐도 비 안 온 날이 겁나 많네
비 온 날이 몇 번 있었나? 비 온다는 예측이 몇 번 맞았나? 이런게 궁금해

우리가 ufunc들을 사용해서 elemental-wise(원소간) 계산을 빨리 하게 됐는데,
그러니 다른 ufunc들을 이용해 데이터들을 주무르며 원소간 비교도 해서 원하는걸 찾을 수 있겠지!
```python
x = np.array([1, 2, 3, 4, 5])
x < 3
# array([True, True, False, False, False], dtype=bool)
x >= 3
x != 3
x == 3
(2 * x) = (x ** 2)
```
[[NumPy 3 Computation on NumPy Arrays. Universal Functions#(3)]]랑 같은 이유로
이것도 '<'나 '>=' 대신 np.less, np.greater_equal로 알아두는게 좋겠지!!

이게 배열에서도 boolean masking이 되더라
```python
rng = np.random.RandomState(0)
x = rng.randint(10, size = (3, 4))
x < 6
"""
array([[True, True, True, True],
	[False, False, True, True],
	[True, True, False, False]], dtype=bool)
"""
```
근데 이 np.random.RandomState(0)가 뭘까?
물론 np.random.seed(0)처럼 무작위 수 다룰 때 필요한거 같긴 한데...

np.random.seed든, np.random.RandomState(0)이든 재현성 있는 전체 난수 흐름을 만들기 위한. 즉, **항상 같은 난수 결과**를 만들기 위한 시드 설정이야!

대신 둘의 차이가 있다면
np.random.seed(0)은 코딩 맨 앞에 나와서 **"전역"** 난수 생성기의 시드를 설정하고,
rng = np.random.RandomState(0)은 rng라는 **"로컬"** 난수 생성기를 새로 만들어서 시드를 설정한대
그래서 로컬의 난수는 np.random.seed(0)과는 독립적으로 작동해!

여담으로 요즘은 np.random.default_rng()를 쓴대.
똑같은건데, 이게 더 빠르고 안정적이라고 권장된대.

참고로 size=(3, 4)는 12개 난수 생성한다는 소리일 뿐.
실제로는 그냥 '벡터화된' **계산이 "한번에" 일어난대**
~~array로 인한 여러 줄이라 계산이 여러 개 일어나는거 아니다~~

## 그 외 명령어들

```python
np.count_nonzero(x < 6)
np.sum(x < 6)
# 세주는 명령어인데,
# sum으로 센다면, 저 조건에 해당하는 애들을 1로 취급하고, 아닌 애들은 0으로 취급

np.sum(x < 6, axis=1)
# array([4, 2, 2]) 이렇게 행 상에서 6보다 작은 애들의 수를 구해줘

np.any(x > 8)
np.all(x < 10)
# 일부나 모두가 충족한다면 True나 False
np.all(x < 8, axis=1)
# array([True, False, True], dtype=bool)로 각 행마다 다 만족하면 True 뜨지
```
여담으로 왜 굳이 파이썬엔 np.count가 아니라
np.count_nonzero가 왜 나오는지 궁금했는데,
boolean 상에서 x < 6을 만족하지 않으면 0이 되니까 0이 아닌걸 세는거야!!
그러니 np.count_nonzero를 하면 True의 개수를 세주는거지!
심지어 지피티가 덧붙였는데, count_ture()라고 하지 않은건
NumPy가 **데이터 타입 중립적**이길 원해서래! bool이든 int든, float이든 똑같이 작동하게!
np.count()자체는 존재하지 않아!

```python
np.sum((inches > 0.5) & (inches < 1))
np.sum(~((inches <= 0.5) | (inches >= 1)))
# &가 and, |가 or. 이런 명령어는 괄호를 정말 잘 신경써야하지
```
&: np.bitwise_and
|: np.bitwise_or
^:np.bitwise_xor
~:np.bitwise_not
명령어로 쓸 때는 이렇게 하자
여담으로 저거 괄호 안 치면 (0.5 & inches)로 해석될거다 ㅋㅋㅋㅋ

```python
x[x < 5]
# 이야 이렇게 하면 5보다 작은 x를 도출해줘. 정확히 말하면 True인 애들만!

rainy = (inches = 0)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print(np.medin(inches[rainy]))
print(np.max(inches[rainy & ~summer]))
# 이런 미친. 조건문 자체를 변수로 쓸 수도 있다...!
```

## and와 or. &와 |

이 두 종류가 쓸 때가 각각 따로 있대
and와 or은 어떤 object 자체의 true나 false를 결정해줘
Python 같은 코딩에선 0 아니면 모두 True지!
```python
bool(42 and 0)
# False
```
즉 각 객체끼리의 비교로 single Boolean evaluation이 일어나

근데 &과 |는 bit 값에서 true나 false를 대조해
말했다시피 0 and 1은 0, 0 or 1은 1인데
```python
bin(42)
# 0b101010
bin(59)
# 0b111011
bin(42 & 59)
# 0b101010
bin(42 | 59)
# 0b111011
```
즉 각 객체의 내용끼리의 비교로 multiple Boolean evaluation이 일어나

그래서 Boolean array로 한다면 같은 결과가 나올거야
```python
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B
# array([True, True, True, False, True, True], dtype=bool)
```
여기서 만약 A or B를 쓴다면, A와 B의 boolean 값이 아닌 
A와 B '자체로' true나 false를 따질거고, 그럼 값이 잘 "하나로" 정의되어 있지 않으니 error가 떠!

결과적으로 우리가 배웠던거에 and를 쓰면. (x > 4) and (x < 8) 같은거 하면 오류 나.

지피티형이 뭘 더 알려준게 있는데,
바로 short-circuit evaluation. 단락 평가야
```python
x = 0
x != 0 and (10 / x > 2)
```
단락 평가는 앞에꺼에서 조건문 결정나면 뒤에 뭔 일이 일어나도 상관 없는거야!
이건 앞에서부터 
근데 &나 |는 양쪽을 모두 계산해서 오류가 생길거야!
여담으로 여기서 and가 x != 0을 괄호로 안 쳐도 제대로 이해하는데,
and나 or은 우선순위가 낮아서, 피연산자 쪽을 전체 표현식으로 먼저 계산한대!
반대로 &와 |는 우선순위가 높은 애들이라 괄호 없으면 망하겠지