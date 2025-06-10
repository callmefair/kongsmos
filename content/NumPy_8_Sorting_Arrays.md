# (1) 비효율적인 sorting들
  
내가 알고리즘 시간에 배웠던 insertion sorts, selection sorts, bubble sorts 같은거 나올건가봐.  
  
## selection sort
  
```python
import numpy as np

def selection_sort(x):
	for i in range(len(x))
		swap = i + np.argmin(x[i]:)
		(x[i], x[swap]) = (x[swap], x[i])
	return x

x = np.array([2, 1, 4, 3, 5])
selection_sort(x)
```  
#### np.argmin(array)
  
array의 최소값의 index를 찾아주는 메소드  
여기선 np.argmin(x\[i]:)라고 했으니까,   
array에서 x\[i]부터 시작하여. x\[i] 부분을 index 0으로 보고, 그 뒤로 가장 작은 애를 찾나봐!  
그리고 놀랍게도.... (x\[i], x\[swap]) = (x\[swap], x\[i]) 만으로 순서가 뒤바뀌나보네...  
  
하지만 알다시피... 이 방식이 O(N\^2)으로 좋진 않지...  
  
## bogosort
  
```python
def bogosort(x):
	while np.any(x[:-1] > x[1:])
		np.random.shuffle(x)
	return x

x = np.array([2, 1, 4, 3, 5])
bogosort(x)
```  
#### np.random.shuffle(array)
  
말해 뭐해... array를 무작위로 셔플한다....  
x\[:-1]은 index 0부터 3까지  
x\[:1]은 index 1부터 4까지  
index 0과 1, index 1과 2, ... 이렇게 둘씩 비교해서 하나라도 오른쪽이 작다면  
array를 냅다 돌려버리는 무책임한 알고리즘이구만  
  
이렇게 index 하나씩 비교해 나가는게 N  
array가 무작위로 돌아가는게 N!라서 O(N x N!)인가봐  
  
당연히 이건 쓰이지 않을거고, NumPy에 더 효율적인게 있겠지  
  
# (2) NumPy의 sort
  
## np.sort(array)
  
np.sort가 걍 메소드 툭 하나 던진건데 겁나 효율이 좋나봐. O(N x log N)이래  
```python
x = np.array([2, 1, 4, 3, 5])
np.sort(x)
# array([1, 2, 3, 4, 5])
```
혹은 그냥
```python
x.sort()
```
  
## np.argsort(array)
  
```python
i = np.argsort(x)
# [1 0 3 2 4]
```
이건 이제 sort를 하려면 그때의 indices 순서를 제공해주는거야  
1번 index가 가장 작다. 0번 index가 두번째로 작다. 이렇게.  
```python
x[i]
# array([1, 2, 3, 4, 5])
```
그래 이렇게 하면 sort된 array가 나오겠네  
  
## 2차원 sorting
  
```python
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))

np.sort(X, axis=0)
# 이렇게 하면 행 오름차순. 즉 각 열마다 sort되어있고.
np.sort(X, axis=1)
# 이렇게 하면 열 오름차순. 즉 각 행마다 sort되어있어.
```
  
# (3) Partitioning
  
## np.partition(array, kth)
  
이건 이제 모든 array를 굳이 sorting하지 않고, 가장 작은 숫자들을 보고싶을 때 쓰는거  
여담으로 뒤에서 ?? 쳐보니까 kth라고 parameter를 말하더라  
나중에 더 알아보니까 k번째로 작은 애는 무조건 k번째로 오게 하고,  
그 앞에는 걔보다 작은 애들. 그 뒤는 걔보다 큰 애들로 해주는거래!  
  
```python
x = np.array([5, 2, 3, 1, 4])
np.partition(x, 2)
# array([1, 2, 3, 5, 4]) 가장 작은 숫자 2가지를 보여준다
```
  
근데 왜 이런게 굳이 필요할까?  
어차피 [[NumPy 8 Sorting Arrays#np.sort(array)|np.sort()]] 하면 가장 작은게 뭔지 알 수 있잖아?  
O(n)의 정도가 다른가??  
  
```python
np.sort??
```
보니까 얘도 kind를 정해서 quicksort나 mergesort나 heapsort 같은걸로 정할 수 있어.  
기본적으로 quicksort로 한다네.  
얘가 옛날에 배웠을 땐 O(N x log N) 정도 평균적으로 가지고, 최악은 O(N\^2)이라던데  
  
```python
np.partition??
```
오 이게 실제로 introselect라고 O(N) 밖에 안해!  
하긴 생각해보면 정렬이 아니라 안의 모든 숫자 다 보고 앞으로 땡겨오기만 하면 되는구나  
  
#### Introselect (Introspective Selection Algorithm)
  
{지피티}  
Introselect라는건 Quicksort와 Heapsort의 하이브리드 알고리즘이래!  
처음엔 Quicksort처럼 빠르게 탐색하다가  
너무 느려지면 Heapsort로 교체해서 최악을 면하게 한대!  
Quicksort로 빠르게 k번째 원소를 찾다가  
너무 길어지면 Heapsort로 전환한다네.  
  
np.partition에선 O(N)이였지만, sortging을 예로 든다면,  
Quicksort가 너무 심해지면 O(N\^2)이 되니까 중간에 Heapsort로 바꿔서 O(N x log N)으로 저점을 높힌다!  
  
#### np.partiton 탐구
  
```python
def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    
    pivot = choose_pivot(arr)
    lows  = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

```
더불어서 np.partition이 어떻게 작동하는지도 물어봤는데,  
1. Quicksort를 하여, 배열의 임의 위치에서 하나의 원소를 고른대.   
보통 중간을 고르고, 이걸 pivot이라고 해  
2. 그리고 왼쪽을 pivot보다 작은 값, 오른쪽을 pivot보다 큰 값으로 옮기나봐  
이 부분 정도는 직접 안 물어봐도 내가 상상의 나래를 펼칠 수 있겠지.  
3. pivot이 위치한 인덱스가 내가 고른거라면 끝나고,  
pivot이 위치한 인덱스가 내가 고른게 아니라면 pivot의 오른쪽이나 왼쪽 array에 새로 pivot을 고를거야  
4. 그렇게 반복하는데, 여기서 오래 걸리면 Heapsort를 한다네?  
  
choose_pivot(arr)은 따로 고르는 알고리즘이 있는거 같고... 내가 보기엔 중간 고를거 같지만  
lows, highs, pivots으로 나눠서, 내가 고른 k랑 pivot의 index를 비교해서 함수를 한번 더 돌리게.  
quickselect 안에서 더 작은 quickselect가 일어나게 해놨네.  
마지막에 else 부분이 만들기 힘들었겠다야....  
  
그래서 heapsort는 어딨는거야??  
```c
if (depth > max_depth) {
    // fallback to heapsort for stability
    heapsort(...)
}
```
지피티형은 그냥 어떻게 작동하는지만 알고 있으래.  
이 안은 매우 복잡하고, 지금의 AI 공부와는 방향이 많이 멀대  
  
## 2차원 partition
  
```python
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
np.partition(X, 2, axis=1)
```
이러면 2번째 열에 각 행마다 2번째 크기인 애들이 전부 배치되겠네!  
  
# (4) Example
  
목표는 특정 점에서 가장 가까운 k개 점 고르기!  
  
#### np.random.rand(m, n)
  
0 ~ 1의 균일분포 난수를 생성하는거래! m행 n열로!  
```python
X = np.random.rand(10, 2)
```
  
```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:, 0], X[:, 1], s=100)
```
여기서 나오는 함수들은 [[NumPy 7 Fancy Indexing#통계와 연결한 예시 (슬슬 통계 나오네)|이걸]] 참조해  
![설명](./images/Pasted%20image%2020250605175015.png)</br>
```python
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape
```
이야 이거 좀 복잡하네... [[NumPy 2 The Basics of NumPy Arrays#reshaping |여기서]] 나온 np.newaxis을 이용하여 3차원 배열로 억지로 만든거 같은데...   
우리가 찾아봤다시피 newaxis는 이 차원에서 그 부분을 1로 만드는 느낌!  
그렇다면 X[:, np.newaxis, :]는 [x좌표, 1, y좌표]

<p><img src="./images/Pasted%20image%2020250605180449.png" alt="설명" style="width: 30%; height: auto;"></p>

X[np.newaxis, :, :]는 [1, x좌표, y좌표] 같은 느낌이려나

<p><img src="./images/Pasted%20image%2020250605180700.png" alt="설명" style="width: 30%; height: auto;"></p>

<p><img src="./images/Pasted%20image%2020250605175556.png" alt="설명" style="width: 30%; height: auto;"></p>

<p><img src="./images/Pasted%20image%2020250605175611.png" alt="설명" style="width: 30%; height: auto;"></p>
내가 [[NumPy 2 The Basics of NumPy Arrays#3차원 array의 concatenate 시험해보기|3차원 배열]] 파헤쳤던걸 생각해보면...  
후자가 가로로 이어져있고, 전자가 세로로 이어져있는 느낌.  
둘이 빼서 broadcasting된 배열을 만든 느낌이야  
<img src="./images/Pasted%20image%2020250605181317.png" alt="설명" width="30%"></br>
결론적으로 x좌표와 y좌표가 broadcasting 되는 모양새지  
  
```python
sq_differences = differences ** 2
dist_sq = sq_differences.sum(-1)
```
이제 저렇게 broadcasting되어서 뺀 값을 제곱해서 더하고 루트하면 거리가 나오겠지?  
꼭 루트를 안 해도 거리 비교는 할 수 있겠지만!  
sq_differences.sum(-1)을 하면 지금 x좌표 y좌표 순으로 뻗어져있는게 axis 2니까  
마지막 axis 기준으로 다 더한다는 느낌인가봐!  
그렇게 더하면 거리의 제곱을 구할 수 있겠지!  
그렇게 해서 10 x 10의 거리 array를 다 구했다!  
  
대신 당연하지만 대각선은 다 0이겠지  
```python
dist_sq.diagonal()
array([ 0., 0. ,0. , ..., 0.])
```
  
```python
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
```
![설명](./images/Pasted%20image%2020250605183036.png)</br>
이제 이렇게 argsort를 해주면 열 오름차순이 되어 각 행마다 sort가 될거야!  
물론 대각선이 다 0일테니, 맨 왼쪽 열이 0 1 2 3 4...로 된걸 볼 수 있지  
  
#### np.argpartition(array, number, axis)
  
얘도 argsort랑 비슷하게, number번째 큰 애를 number번째 index에 놓고 싶은데,  
그렇게 하기 위한 indices들을 array로 낸거  
```python
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
```
우리는 각 점마다 가장 가까운 두개의 점을 찾고 싶고,  
어차피 제일 가까운건 자기 자신일테니, K + 1로 3개의 가장 가까운 점을 구하자  
```python
plt.scatter(X[:, 0], X[:, 1], s=100)
K = 2

for i in range(X.shape[0]):
	for j in nearest_partition[i, :K+1]:
		plt.plot(*zip(X[j], X[i]), color='black')
```
![설명](./images/Pasted%20image%2020250605184739.png)</br>
대충 코드 써보니 선을 긋는 코딩을 한거 같은데,  
아무래도 한 점당 점 3개까지 선을 그어서, 본인에서 본인으로 가는 선은 사라지는 느낌인거겠지?  
당연히 어떤 점에서 가장 가까운 점이 있다고 해도,  
그 가까운 점에서 가까운 점이 처음 점은 아닐거야.  
그래서 한 점에 선이 저렇게 2개보다 많은거고 ㅇㅇ  
  
여하튼 이게 루프 돌리는 것보단 훨씬 빠른가봐  
이걸 size에 대해 [[단어장#^agnostic|agnostic]]이라고 표현했네  
끝으로 O(N)에 대한 설명 하는데... 이건 뭐 수학적으로나 코딩적으로나 알건 아니까....  
  
# 지피티 연습문제
  
어떤 반의 학생 10명의 수학 점수가 다음과 같이 주어져 있다:  
```python
scores = np.array([78, 95, 62, 88, 70, 99, 84, 73, 91, 67])
names = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
```
ㄱ. for, if 문 사용 금지  
ㄴ. print(sorted) 등 단순한 문법 금지  
ㄷ. sorting arrays에서만 배운 기능만 활용  
  
```python
# 1. 점수가 "높은" 순서대로 학생들의 이름을 정렬하라.
indices = np.argsort(scores)
print(scores[indices[::-1]])
print(names[indices[::-1]])

# 2. 점수 상위 3명의 이름과 점수를 출력하라.
indices2 = np.argpartition(scores, -3)
# 훗 argpartition을 검색하다가 partition에 index를 마이너스로 놓으면 큰 애를 뽑는단걸 알게됐지
print(scores[indices2[-3:]])
print(names[indices2[-3:]])

# 3. 가장 낮은 점수를 받은 학생의 이름과 점수를 출력하라.
print(scores[indices[0]])
print(names[indices[0]])
```
2번 문제는 1번 문제에서 바로 풀 수 있긴 하지만...  
argpartition 배운 김에 함 써먹어보자  
파이썬만 그런진 모르겠는데, 이 index 자유자재로 써먹을 수 있는게 확실히 매력인거 같아.  
