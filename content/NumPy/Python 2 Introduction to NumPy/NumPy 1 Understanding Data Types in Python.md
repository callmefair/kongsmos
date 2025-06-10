
### 정적 동적 타입 언어

**statically typed language**: 정적 타입 언어
C나 Java 같은게 있어
<u>각각의 변수가 명시적으로(explicitly) 선언되어야 해</u>

**dynamically typed language**: 동적 타입 언어
파이썬 같은게 있어
이런 <u>명시적 선언이 필요가 없나봐!</u>

C는
```C
int result = 0;
for(int i = 0; i < 100; i++){
	result += i;
}
```

파이썬은
```python
result = 0
for i in range(100):
	result += i
```
음 대표적으로 result나 i에 int 같은게 안보이는게 큰 차이네!
어떤 데이터든 어떤 변수에 집어넣어도 상관 없다
심지어 x에다가 integer 집어넣다가 string 집어넣어도!
이런 유연성이 오히려 각 변수마다 그 값 이상의 무언가가 존재한다는건가봐

*그 type에 추가적인 정보가 있다?*

### Python에서의 object란?

우리가 파이썬에서 x = 10000이라고 할 때, 
x가 그저 정수인게 아니래!
파이썬이 애초에 C에서 파생된건진 몰라도
이 x가 C의 어떤 복합적인 구조로 pointer를 찍는거 같아
소스 코드를 보면,
```python
struct _longobject {
    long ob_refcnt;
    PyTypeObject *ob_type;
    size_t ob_size;
    long ob_digit[1];
};
```
~~이건 그냥 이렇게 생겼다는거 보려고 직접 치지 않고 복붙했어~~
*이 각각을 설명해줬는데, 중요할지는 이따가 다 보고 결정하자*
아무튼 정보가 많다.
C는 integer 하나 만들면, integer 값을 가진 bytes만 있을 뿐인데,
파이썬은 파이썬 object의 정보를 담은 memory에 pointer가 있지. 그 안에 integer bytes도 있고.

아무튼 안에선 복잡한 과정이 이루어지지만, 이로 인해 파이썬은 코딩을 자유롭게 동적으로 할 수 있어.
**이런 추가 정보들을 담기 때문에 비용은 많이 들어**
특히 여러 개의 객체를 한데 모아놓은 리스트나 딕셔너리에선 이 비용이 훨씬 많이 든다네;;
변수 무더기!!

### Python에서 object가 많다면

많은 object를 가진 데이터를 다룬다면 무슨 일이 일어날까?
Python은 동적 타입이라 이런 [[단어장#^heterogenous]]한 리스트도 만들 수 있대
```python
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]
```
하면 \[bool, str, float, int\] 같은거 나오나봐
근데 이렇게 하면 리스트 안에 있는 하나씩마다 위의 struct처럼 정보가 필요하겠지. 겁나 [[단어장#^redundant]]해

![[Pasted image 20250516153319.png]]
보통 Numpy array는 [[단어장#^implementation]] 단계에서, 데이터 블럭으로 [[단어장#^contiguous]]포인터를 찍어. \[1\]\[2\] 이렇게 연속적으로 찍지. 
즉, 포인터가 하나야. 

근데 Python 리스트는 하나의 포인터로 block of pointers로 찍고, 그 block of pointers에 있는 포인터는 위의 정보 많던(ref count, type info 등) 각각의 \_longobject 같은 데로 이어지나봐.
bool, str, float 같은게 가능한 이유도 그것 때문! 
**연속적인 데이터가 아니라 리스트 안의 데이터들을 각자 다른 데서 불러와.** 참조를 드럽게 많이 하나봐

Python은 유연하지만, 디지게 느릴거다

### ~~Integer Array?~~

```python
import array
L = list(range(10))
A = array.array('i', L)
A
```
이렇게 하면 integer type의 리스트를 만들 수 있나봐
근데 NumPy에 나오는 ndarray가 더 유용하대. 얘로 하면 좋은 명령어를 쓸 수 있나봐

### NumPy Array

```python
import numpy as np
np.array([3.14, 4, 2, 3])
```
아무튼 우리는 numpy array를 써서 같은 type 애들로 구성된거 만들건데,
이렇게 다양한 type으로 np.array하면 array(\[3.14, 4.   ,  2.   ,  3.    ])
이런 식으로 자동으로 type 조정이 되나봐

```python
np.array([1, 2, 3, 4], dtype = 'float32')
```
혹은 아예 dtype이란 걸로로 type을 정할 수도 있고,
type 종류도 적혀있긴 하던데, 그냥 필요할때 보자


### NumPy Array Function

```python
np.zeros(10, dtype = int)
np.ones((3, 5), dtype = float)
# 3행 5열 float type 1로 가득 채우기
np.full((3, 5), 3.14)
np.arange(0, 20, 2)
# 0 이상 20 미만 2씩 늘리기
np.linspace(0, 1, 5)
# 0 이상 1 이하 5분할 하기
np.random.random(0, 1)
# random 2개 나오면 0하고 1 사이로 잡아주는듯
np.random.normal(0, 1, (3, 3))
# normal로 하면 범위 정할 수 있고
np.random.randint(0, 10, (3, 3))
# randint는 random integer의 준말이겠지. 3행 3열 0이상 10미만 랜덤 숫자
np.eye(3)
# 놀랍게도 identity matrix도 만들어준다
np.empty(3)
# 크기가 3인 배열을 만들되, 초기화는 하지 않는다
# 배열을 빨리 만들고, 나중에 값을 채우거나 할 때 좋아. 그냥 메모리 공간만 확보!
```
Creating arrays [[단어장#^fromscratch]]

## 맨 끝의 절념

Python은 동적 타입이라 겁나 유연한 애인데, 그만큼 돈과 시간을 많이 잡아먹어
근데 NumPy로 배열 같은거 만들어서 쓰려면 어차피 같은 datatype일텐데 왜 C를 안 쓰는거지?
특유의 유연함이 빛을 발하는 상황이 오려나?
아래를 자세히 보니 NumPy가 C에서 만들어진 애래.
그렇다면 왜 Python에서? 물론 Python이 AI 배울 때는 짱이라지만...

### {지피티} NumPy의 정체

사실 Python은 겉껍데기다. 
내부는 C고, 외부는 Python으로 감싸져있는 구조래
배열 연산 같은건 C 수준에서 진행돼서, np.sum() 같은게 Python보다 몇 십배 빠르대!
그래서 위에서 ndarray가 array.array보다 빠르다고 했는데,
**이런 np. 뭐시기로 만들어진 array가 ndarray datatype인거야!**
대신? 조작 인터페이스. 
슬라이싱 인덱싱 같은건 **Python 문법이라서 쓰기 아주 쉽단 장점이 있어**
둘이 합쳐서 빠르고, 직관적이게!
