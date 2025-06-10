Vectorized operation을 좀 써서 효율적으로 array 계산해보려나 보다

```python
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
	output = np.empty(len(values))
	for i in range(len(values)):
		output[i] = 1.0 / values[i]
	return output

values = np.random.randint(1, 10, size = 5)
compute_reciprocals(values)

# 별로 이상해보일건 없는데, 이거 엄청 큰 리스트로 하면 시간 겁나 길어진대

big_array = np.random.randint(1, 100, size = 1000000)
%timeit compute_reciprocals(big_array)
```

%timeit은 전에 넘어갔었지만 시간 재주는 애
아무튼 이런 계산을 하는데 무려 몇 초가 걸려! 코딩 치곤 느리지
이런거 역수 구할 때 파이썬은 그 object의 type과 그 type에 적절한 function이 이용되었는지를 보면서 함수를 동적으로 찾아서 실행해.
각각의 x에 대해 다르게 정의된 나눗셈 연산 함수를 일일이 찾아서 호출하지

근데 **컴파일된 코드**에서는 그 값이 어떤 타입인지 미리 알 수 있어
그래서 이미 적절한 연산 방법이 정해져 있고, 훨씬 더 빠르게 계산 가능해
컴파일은 전체 코드를 미리 번역한 후 실행하는 애래
위의 함수로 따지면
```C
double reciprocal(double x) {
	return 1.0 / x
}
```
이런 식으로 double로 이미 고정되어 있는거지!! 그래서 python보다 빨라

```python
print(1.0 / values)
# 근데 만약 이렇게 했다면 어떨까?? 일단 코딩 결과는 같아
%timeit (1.0 / big_array)
```
왠진 모르겠지만 확연히 빨라진게 느껴진다
뭔가 계산이 벡터처럼 이루어졌단걸 알 수 있어!
저 계산에 벡터값을 끼워넣은 느낌!!

## (1)

이런 vectorized operaton들은 **ufuncs**이란걸로 작동하나봐
NumPy arrays에서 반복되는 명령을 빠르게 실행하기 위한거야!
```python
np.arange(5) / np.arange(1, 6)
```
이런 것도 가능하지! 0, 0.5, 2/3, 0.75, 0.8 같은거 나오겠네

```python
x = np.arange(9).reshape((3, 3))
2 ** x
```
이렇게 하면 reshape된 array에서 각 값의 2의 거듭제곱이 나오는거고.
이러면 loop시킨 것보다 편리하고 빠르네

이런 ufuncs는 사칙연산, 나머지, 마이너스, 제곱 그런거 죄다 x ** 2 이런 식으로 명령하면 끝!
당연하지만 -(0.5 * x + 1) ** 2 같은 이상한 것도 잘 해줘!
더하기 말고 np.add(x, 2) 이런 것도 되지만, 그래도 편한게 편한거지
삼각함수도 가능해. np.arcsin(x) 이런 것도 가능!
여담으로 파이는 np.pi네
np.exp(x), np.power(3, x)
np.log(x), np.log2(x)

뭘 위해 있는지는 모르겠지만
e^x - 1로 np.expm1(x)
log(1 + x)로 np.log1p(x)
x가 너무 작을 때 사용하는 느낌이라네

## (2)

```python
from scipy import special
special.gamma(x)
special.gammaln(x)
special.beta(x, 2)
special.erf(x)
special.erfc(x)
special.erfinv(x)
```
...뭔지도 모르겠는 것들인데 필요하면 내가 알아서 찾아보겠지
수학교양도서에나 본 것 같은 유명한 함수들이겠지

## (3)

```python
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out = y)

z = np.zeros(10)
np.power(2, x, out = z[::2])
% 이런 것도 가능하구나... 짝수번째마다 결과 나오게 하기
```
이렇게 하면 ufuncs의 결과를 원하는 array에 저장해놓을 수 있겠지
이래서 x + 2 이런거 말고 명령어도 기억해야 하나봐!

```python
x = np.arange(1, 6)
np.add.reduce(x)
np.multiply.reduce(x)
np.add.accumulate(x)
# array([1, 3, 6, 10, 15])
np.multiply.accumulate(x)
```
reduce는 결과가 하나만 나올 때까지 계속 연산을 반복하는 method래!
accumulate는 각 결과를 하나의 array에 놓는 것!
다른 것도 있는데 뒤에서 보는걸로

## (4)

```python
x = np.arange(1, 6)
np.multiply.outer(x, x)
```
이러면 x의 구구단 테이블을 만들어주더라
    1 2 3 4 5
 1
 2
 3
 4
 5
이렇게 다 곱해진거!

