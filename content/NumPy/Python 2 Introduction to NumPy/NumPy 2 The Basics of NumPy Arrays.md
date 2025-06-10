# NumPy Array Attribute

*코딩에서 attribute는 property랑 같이 속성인데, property와 달리 attribute는 동적에서 쓰는 말인가봐?*
```python
import numpy as np
np.random.seed(0)

x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))
# 각각 dimension이 1, 2, 3인 array야

print(x3.ndim) # 3
print(x3.shape) # (3, 4, 5)
print(x3.size) # 3 x 4 x 5로 해서 60
print(x3.dtype)
print(x3.itemsize) # 8 bytes
print(x3.nbytes) # 480 bytes
```
<u>이런 .뒤에뭐붙는거</u> 이게 **attribute**라고 하나봐^attribute

```python
x1[4]
x2[0, 1]
```
이런걸 어떤 데이터에 접근하는 accessing이라고 표현하는 것 같고,

## Slicing

slicing은 array 잘라서 일부만 남기는거. subarray라고 볼 수 있을지도.
```python
x = np.arange(10)
x[:5] # index 5까지 자르기
x[5:] # index 5 이후로 자르기
x[1::2] # index 1부터 해서 2씩 자르기
```
x\[start:stop:step] 같은 느낌으로 보면 될거 같아
step에는 -1 같은 것도 넣을 수 있고.

```python
x2[:2, :3] # 2행 3열로 잘라준다
x2[:3, ::2] # 3행을 하되, 열은 처음부터 포함해서 2씩 자르자
x2[::-1, ::-1] # 이렇게 하면 아예 거꾸로 만들 수도 있겠다
```
2차원 array에서도 재밌는 활용법을 만들 수 있지

slicing하고 accessing을 합쳐 쓸 수도 있어
```python
print(x2[:, 0]) # 1열이 나와
print(x2[0, :]) # 1행이 나와
# 사실 그냥 print(x2[0])을 해도 1행만 나오더라고

x2_sub = x2[:2, :2]
print(x2_sub)
# subarray를 지정할 수도 있고
```

```python
x2_sub[0, 0] = 99
print(x2_sub)
print(x2)
```
**근데 웃긴건! x2_sub의 (0, 0)을 바꿨는데, x2도 (0, 0)이 바뀐다는거지**
*난 이게 오히려 안 좋은 기능인줄 알았는데, 따로 데이터 더 안잡아먹고 복사할 수 있어서 좋은거래*

```python
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42
```
이렇게 하면 원본이 안 바뀐다네!

### reshaping

```python
grid = np.arange(1, 10).reshape((3, 3))
print(grid)
```
이건 1차원 array를 3 x 3에 집어넣은 느낌이군. 이걸 reshaping이라고 해
근데 이거 하려면 전과 후의 크기가 맞아야 한대!
그리고 reshape는 ~~원래 배열의 데이터를 새로 복사하지 않고,~~
**배열을 바꾸는 "보기". view만 바꾼대**
근데 메모리 버퍼가 연속적이지 않으면, 메모리 상의 값들이 떨어져서 저장되어 있다면,
<u>reshape가 view를 못 만들고, 진짜로 복사할 수도 있대</u>
**지피티형이 말하기를, AI에서 딥러닝 텐서를 다룰 때 reshape를 겁나 많이 쓴다네!**

```python
x = np.array([1, 2, 3])
x.reshape((1, 3)) # 1행 3열로 reshape 해라
x[np.newaxis, :] # newaxis. 말 그대로 새로운 축으로 만든다
x[:, np.newaxis]
```
옛날에 공부할 때 이거랑 비슷한걸 본 것 같아. x\[:] 이런 식으로 표현하기도 했어.
그냥 np.newaxis 자체가 새로운 축에서의 1을 의미하는 것도 같아

### concatenate

```python
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y]) # array 합치기! % array([1, 2, 3, 3, 2, 1])
z = [99, 99, 99]
np.concatenate([x, y, z])
```
[[4. String#concatenate]]의 NumPy 형식인거지

```python
grid = np.array([1, 2, 3],
				[4, 5, 6])
np.concatenate([grid, grid]) 
"""
이렇게 하면 array([[1, 2, 3],
				[4, 5, 6],
				[1, 2, 3],
				[4, 5, 6]])
첫번째 axis. 즉 행에 이어 붙이는거
사실 1차원 array 예시도 이 방식과 같다고 볼 수 있지
"""

np.concatenate([grid, grid], axis = 1)
"""
이렇게 하면 array([[1, 2, 3, 1, 2, 3]
				[4, 5, 6, 4, 5, 6]])
index 1. 두번째 axis. 즉 열에 이어 붙이는거
"""
```
### 3차원 array의 concatenate 시험해보기

![[Pasted image 20250516171644.png]] 
과연 3차원에 axis는 어떻게 작동할까?
![[Pasted image 20250516172440.png]]

![[Pasted image 20250516171713.png]]
![[Pasted image 20250516172655.png]]

![[Pasted image 20250516171729.png]]
![[Pasted image 20250516172747.png]]

![[Pasted image 20250516171741.png]]
![[Pasted image 20250516173013.png]]
그래 뭐 얘까진 어떻게든 설명이 되는 모습이네
아무튼? axis는 각 축으로 옆에 붙이는거다! axis = 1의 의미는 그거였다!


### vstack, hstack

```python
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
				[6, 5, 4]])
np.vstack([x, grid])
"""
vertically stack
array([[1, 2, 3],
	[9, 8, 7],
	[6, 5, 4]])
"""

np.hstack([grid, y])
# 이거는 뭐 axis = 1 같은거겠지
```

### splitting, vsplit, hsplit

```python
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
# index 3 전, 5 전에서 잘라라
print(x1, x2, x3)
```
이게 splitting

```python
grid = np.arange(16).reshape((4, 4))
grid
"""
array([[0, 1, 2, 3],
		[4, 5, 6, 7],
		[8, 9, 10, 11],
		[12, 13, 14, 15]])
"""
upper, lower = np.vsplit(grid, [2])
# 이러면 또 0~7까지 하나, 8~15까지로 하나로 나뉘지

left, right = np.hsplit(grid, [2])
# 이건 0 1 4 5 8 9 12 13으로 나뉘는거야
```

## 맨 끝의 절념

