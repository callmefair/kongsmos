universal function들은 명령어를 vectorize 해서 느린 Python loop들을 굳이 안 써도 되게 했어
근데 이렇게 명령어를 vectorize하는 다른 방법이 있지. Broadcasting

일단 Broadcasting은 이산. 즉 사칙연산 ufuncs를 적용하는 규칙이라고 하는데... 
```python
import numpy as np
a = np.array([0, 1, 2])
a + 5
# array([5, 6, 7])
```
이렇게 다른 size더라도 덧셈을 수행할 수 있지
이런 5가 \[5, 5, 5]가 되는걸 실제로 값으로써 자리를 차지하지 않는 duplication으로 볼 수 있지만,
이걸 하나의 '생각 모델'(mental model)로 받아들일 수도 있어!
마치 저 5가 \[5, 5, 5]처럼 같은 길이로 늘어난 것처럼 생각할 수 있는거지
실제로 일어나진 않지만 그렇게 상상하면 개념이 더 잘 잡힌다

물론 array에서도 적용되지
```python
M = np.ones((3, 3))
M + a
"""
array([[1., 2., 3.],
		[1., 2., 3.],
		[1., 2., 3.]])
"""
```

broadcasting이 두 array에서 일어나는 경우도 있어
```python
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b
"""
array([[0., 1., 2.],
		[1., 2., 3.],
		[2., 3., 4.]])
"""
```
이야 1행 3열하고 3행 1열하고 합치니까 3행 3열로 계산이 되게 하네

## (1)

이런 broadcasting에는 엄격한 규칙이 있대
1. 두 array가 dimension이 다르다면, 주로 낮은 dimension 쪽이 높은 쪽으로 덮여진다고 해
여기서 broadcasting은 **left padding**한다고 하는데!
만약 a + b를 해서 b가 a의 dimension을 맞춰준다고 한다면 말야...
~~여기서 b가 a로  차원을 맞춰준다는 소리가 아니라~~
차원 번호. 이 Numpy에선 행이 1번, 열이 2번인데
이 순서에서 맞춰준다고 얘기하는거야! 
이 Numpy는 오른쪽에서 왼쪽으로 padding 해서 차원 비교를 한다!
차원이 모자란 배열은 **"왼쪽부터"** 1을 채워넣는다!
```python
a = np.array([[1],
			[2],
			[3]])
b = np.array([1, 2, 3])
"""
여기서 b가 1차원 배열인거잖아?
그러면 b의 현재 dimension 상 shape를 (3,)이라고 할 수 있는데,
우리는 dimension을 맞춰야 하니까 가상으로 padding 하여 (1, 3)이라고 하는거지
"""
```

1-1. 근데 여기서 굳이 right padding을 하고 싶다면,
이 책에선 굳이 머리 쓰지말고, a = a\[np.newaxis, :]를 해서 shape를 (3, 1)로 바꿀 수 있겠대
```python
a.shape
a[np.newaxis, :].shape
```
그리고 애초에 이렇게 shape를 찾을 수도 있어!

2. 두 array가 shape가 어떤 dimension에서도 맞지 않다면,
the array with shape equal to 1 in that dimension
그 차원에서 배열의 크기가 1인 배열은: 그 차원의 값이 자동으로 반복돼서 맞춰진다.
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape: (2, 3)

B = np.array([[10],           # shape: (2, 1)
              [20]])
A + B
```
B의 두 번째 차원. 열의 크기가 1이니까 열의 크기를 자동으로 3까지 늘리는거지!

3. size들의 dimension이 맞지도 않고, 1도 아니라면 error 발생!
아쉽게도 최소공배수는 일어나지 않는 모습이다.
3 x 2 행렬 + 1 x 3 행렬이면 바로 오류 생기는거야

이 모든 broadcasting은 ufunc이면 다 적용된대!
```python
np.logaddexp(A, B)
# 여담으로 이 ufunc은 log(exp(a) + exp(b)) 식으로 작동
# 이제 결론이 우리가 아는 A + B shape인데 저 결과값이 나오겠지!
```

## EX

```python
X = np.random.random((10, 3))
Xmean = X.mean(0)
Xmean
# 이러면 대충 10행의 평균값이 각 열마다 나오겠지
X_centered = X - Xmean
# 이렇게 broadcasting하고,
X_centered.mean(0)
# 이거 하면 0에 가깝게 나오면 잘 한거지
```

```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar();
```
이것도 오랜만에 보네. 수학과에서 할법한 import야