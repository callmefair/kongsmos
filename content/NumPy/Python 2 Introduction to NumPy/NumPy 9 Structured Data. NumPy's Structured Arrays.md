아무래도 여긴 챕터 3를 위한 초석을 닦는 것 같네...
# (1) 다양한 종류의 data들 묶기

```python
import numpy as np
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
```
우리가 다양한 종류의 data를 저장한다고 하면 이럴 수 있을텐데,  
문제는 이게 data들끼리 전혀 연결이 안되지... 같은거라곤 index 뿐이네  
  
근데 우리가 저번에 np.zeros(number, dtype)을 배운적 있지?  
이것마저도 이 data type에 맞게 유연하게 쓸 수 있더라고?  
```python
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
							'formats':('U10', 'i4', 'f8')})
print(data.dtype)
# [('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]
```
이 <U10 같은게 뭐인고 하니... Unicode string of maximum length 10이래
i4의 i는 integer, f8의 f는 float이고!

여기서 이런 명령어를 치면?
```python
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
# [('Alice', 25, 55.0), ...]
```
이런 식으로 각 dtype에 우리가 넣고자 하는 data들이 들어가고,
한 리스트의 tuple로 저장이 되는거 같아

```python
data['name']
# array(['Alice', 'Bob', 'Cathy', 'Doug'], 
#      dtype='<U10')
data[0]
data[-1]['name']
```
이런게 다 가능해 이제!

심지어는
```python
data[data['age'] < 30]['name']
# array(['Alice', 'Doug'], 
#      dtype='<U10')
```
이런 식의 boolean형 명령어도 편하게 가능하지! 유연성 하나는 엄청나다
이게 나중에 Pandas에서 dataframe이란거 배울 때 쓰이나봐. 걔네들만의 더 복잡한 명령어로!

# (2) Structured Arrays

Structured data들을 묶는걸 배웠다면,
애초에 data들이 묶인 Structured Array를 만드는 방법도 알아야겠지
```python
np.dtype({'names':('name', 'age', 'weight'),
		'formats':('U10', 'i4', 'f8')})
```
이렇게 하면 NumPy에서 data type을 만들 수 있나봐
뭐 간단하게 할거면 U10 이런게 낫고, 파이썬식의 data type인 (np.str_, 10)이나 np.float32도 가능하다네
```python
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
```
물론 이렇게 tuple 식으로 만들 수도 있지
아니면 굳이 name, age 같은 이름을 안 짓더라도?
```python
np.dtype('S10,i4,f8')
# dtype([('f0', 'S10'), ('f1', 'i4'), ('f2', '<f8')])
```
이제 와서야 얘기하지만 \< 같은건 little endian이라는데, ~~컴퓨터가 정렬하는 숫자 구성 바이트가 더 적다는 뜻인가봐!~~
근데 왜 어떤건 \<가 붙고, 어떤건 안 붙는걸까?

### little endian

{지피티}
일단 <는 NumPy가 내부적으로 바이트 순서를 표시하는 방식이래
컴퓨터는 숫자를 메모리에 저장할 때 왼쪽부터 저장할지, 오른쪽부터 저장할지 결정해야 한다는데,
little endian. \<는 작은 바이트부터 저장한다는 소리야
아무래도 숫자를 이진법으로 표현한다면 1, 2, 4, 8... 의 순서로 저장한다는 소리 아닐까?
숫자 같은건 값을 의미하다 보니까 \<나 \>가 중요하고,
boolean 같은건 순서 의미가 없어서 |라는게 보통 앞에 붙는대!

## 2차원 type

```python
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
```
자 위에서 배운걸 토대로 생각해보면,
이 tp라는 type은 i8짜리 id랑 f8짜리 mat를 가지고 있는데, 
이 mat는 이제 3x3 행렬인거지
그럼 X는 (integer, float matrix)의 원소를 갖겠네

# (3) 그래서 우리 이거 왜 씀?

근데 우리 dictionary나 Python만의 유연한 array도 있는데,
왜 굳이 이런 새로운 structured data를 만드는 법을 배워야 할까?
{지피티 해석}
Python은 우리가 [[NumPy 1 Understanding Data Types in Python#Python에서의 object란?|여기서]] 배웠던거처럼 C의 struct에 매핑이 되는데,
이 np.dtype이 C의 struct에 바로 매핑해서 더 빠르게 읽을 수 있나봐!
실제로 우리가 여러 data의 array에서 C로 매핑할 때 겁나 상황이 복잡해지는걸 봤잖아!
심지어는 dictionary는 C랑 연동되지도 않는대.
이게 아무래도 각 data마다 type도 같이 저장되다 보니까, C나 fortran에서 직접 읽을 수가 있대!
한 마디로 데이터 쉽게 연동하기 위함이다~
{지피티}
전에 배운 struct랑 관련이 있냐고 더 질문해봤는데,
옛날에 배운 struct에 따르면
우리가 Python 객체를 만들면, 그 값만 저장하는게 아니라 struct로 따로 정의 및 구현이 되는거였고
우리가 직접 만든 np.dtype은 아예 처음부터 우리가 C에서 struct를 만들어서 맞춰서 넣은 느낌이야! 우리가 직접 데이터를 정의했다!


# (4) np.recarray

우리가 배운 structured data랑 비슷하게 NumPy에 또 이런게 있는데,
```python
data['age']
```
우리가 기존에 이렇게 data를 봤다면,
```python
data_rec = data.view(np.recarray)
data_rec.age
```
np.recarray는 dictionary key 말고, attribute로 접근할 수 있지!
아니 근데 np.recarray 뭔데 갑자기 튀어나와서, 정의도 안하고 바로 저렇게 쓸 수 있는거냐?
저 recarray라는 놈은 structured array를 

근데? 겁나게 느려
data\['age'] >>>> data_rec\['age'] > data_rec.age 순으로 점점 느려져
활용에 따라 데이터는 적은데, 글자 적게 치고 싶으면 후자
데이터 겁나 많으면 전자를 쓰는게 좋을듯??