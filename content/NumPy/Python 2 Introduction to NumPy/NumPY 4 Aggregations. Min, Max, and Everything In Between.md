aggregate 집계하다
우리가 겁나 많은 데이터 보면 평균, 분산 내거나 max, min 같은거 구하겠지
NumPy에서 이런거 빨리 만드는 기능 구하는듯

```python
import numpy as np
big_array = np.random.random(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)
```
후자의 NumPy 버전이 컴파일되어 있어서 훨씬 빠르게 진행돼
여기서 sum과 np.sum이 다르단걸 알아둬야 한대.
지금 당장은 이유를 알 수 없지만, np.sum이 여러개의 array dimension을 인식하고 있다네?

마찬가지로 min, max보다
np.min, np.max가 훨씬 빠르더라고.
```python
print(big_array.min(), big_array.max(), big_array.sum())
```
글자수를 더 줄이기 위해 이렇게도 쓴다고 해
**syntax**란 문법. 쓰기 방식
어떤 언어에서 문장을 어떻게 써야 올바른 코드로 해석되는지에 대한 규칙.
np.sum(array)는 함수형 syntax
array.sum()은 객체 method syntax래. 객체지향적이지

**객체지향적**??
데이터와 그 데이터를 다루는 함수를 하나의 단위(객체)로 묶어서 프로그래밍 하는거래

```python
numbers = [1, 2, 3]

# 함수형 스타일
len(numbers)

# 객체지향 스타일
numbers.append(4)
numbers.sort()
```
numbers라는 리스트 '객체'에
append나 sort 같은 '기능'. method가 "같이 묶여있는 것"
이게 객체지향적인거래

반대로 절차지향은 순서대로 명령을 처리하는 방식이래
객체지향은 그 데이터에 애초에 기능이 묶여있어서, 그 데이터 자체. 객체가 스스로 행동하는거고!
동적 타입 언어에선 이런 method 만드는게 굉장히 유연하게 구현된다고도 하네

아무튼 다시 원래 얘기로 돌아와서...
multi dimension이여도 aggregates를 쓸 수 있어
```python
M = np.random.random(3, 4)
M.sum()
M.min(axis = 0)
# 이러면 각 열에서 가장 작은 애 골라주고
M.min(axis = 1)
# 이러면 각 행에서 가장 작은 애 골라줘
```
아무래도 [[NumPy 2 The Basics of NumPy Arrays#(3)]]에서 본 
axis에 따라 array에 데이터가 추가되는 방식을 생각해보면 
어떤게 행이고 어떤게 렬인지 알 수 있겠지 

# (1)

근데 대부분 aggregates들이 NaN-safe한 버전이 있대!
**NaN** = Not a Number. 수학적으로 정의되지 않은 값.
이런 NaN이 포함된 데이터라도 연산이 깨지지 않게 잘 계산해주는 aggregates들이 있는거야!
```python
x = np.array([1, 2, np.nan, 4])
np.nansum(x)
```
이렇게 해도 NaN 무시하고 나머지 값만 더해주는거지
현실 데이터는 그렇게 깨끗하지 않거든!!
여담으로 Pandas는 NaN을 무시하고 처리한대
