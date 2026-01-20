+++
date = '2026-01-21T03:00:46+09:00'
draft = false
math = true
title = 'Royden Ch2-4. Measureable set의 속껍질과 겉껍질 근사'
+++

# 🐚1. measurable set의 속껍질과 겉껍질

![](https://velog.velcdn.com/images/7kong/post/8f7320ab-d6e4-49a4-a06f-6895cdcaa470/image.jpg)

속껍질과 겉껍질 근사.<br>
우리가 outer measure를 정의할 때 이미 이런 성질을 갖고 있어보였어.

We define the **outer measure** of $A$ , $m^*(A)$ , 
to be the infimum of all such sums, that is
$$m^*(A) = \mathsf{inf} \{\sum^\infty_{k=1} l(I_k) \; | \; A \subseteq \bigcup ^\infty _{k=1} I_k\}$$

저번에도 비슷하게 말했던 것 같은데,
$l(I_k)$들이 **겉껍질**처럼 덮어주고, $\mathsf{inf}$가 **속껍질**처럼 최대가 되어주도록 밀어주지.

이번에 그것과 비슷하게 inner approximation과 outer approximation이란걸 배울건가봐

## 1-1. Excision Property

일단 이번 챕터에서 제일 많이 쓰이는 정리.
직역하면 절단 성질이라는데, 
*구글에 대놓고 치면 이상하게 대수적 위상수학이 먼저 언급된다?
언젠간 한번 더 마주할 수도 있겠다*

>If $A$ is a meausrable set of finite outer measure that is contained in $B$, then
>$$m^*(B \sim A) = m^*(B) - m^*(A)$$

$$m^*(B) = m^*(B \cap A) + m^*(B \cap A^C) = m^*(A) + m^*(B \sim A)$$
뭐 애초에 measurable 정의가 이거고,
$A$가 finite하다니까 우리가 찾는 property의 식이 완성되지.

## 1-2. 겉껍질과 속껍질이 되어줄 친구들

>Let $E$ be any set of real numbers. Then each of the following four assertions is
>equivalent to the measurability of $E$. 
>(Outer Approximation by Open Sets and $G_{\delta}$ Sets)
>(i) For each $\epsilon > 0$, there is an open set $\mathcal{O}$ containing $E$ for which $m^*(\mathcal{O} \sim E) < \epsilon$,
>(ii) There is a $G_{\delta}$ set $G$ containing $E$ for which $m^*(G \sim E) = 0$
>
>(Inner Approximation by Closed Sets and $F_{\sigma}$ Sets)
>(iii) For each $\epsilon > 0$, there is an closed set $\mathcal{F}$ contained in $E$ for which $m^*(E \sim \mathcal{F}) < \epsilon$,
>(iv) There is a $F_{\sigma}$ set $F$ contained in $E$ for which $m^*(E \sim F) = 0$

겉껍질이 될 수 있는 $\mathcal{O}$와 $G_\delta$ set, 속껍질이 될 수 있는 $\mathcal{F}$과 $F_\sigma$ set의 존재성 얘기야.
증명 일부는 지금까지 나왔던 테크닉이랑 비슷하고, 특별하지 않아서 별로 적고싶진 않다.

일단 증명 방식은 measurablity of $E$ => (i) => (ii) => measurablity of $E$의 순서대로 해!
### measurable => (i)

$E$가 measurable하다고 하고, $I_k$들이 $E$ 감싼다고 해보자고.
"$\mathcal{O}=\bigcup _{k=1}^\infty I_k$"라고 하면, $\mathcal{O}$는 $E$를 감싸고, 
$\epsilon$으로 부등호 역전시키는 식을 만들 수 있어

$E$가 finite measure라면 Excision Property에 의해 (i)을 증명할 수 있어
근데 $m^*(E) = \infty$라면 어떨까?

$E$를 $\{E_k\}^\infty _{k=1}$로 쪼개서 disjoint union of countable collection으로 하고,
이를 open set $\mathcal{O}_k$가 $m^*(\mathcal{O}_k \sim E_k) < \epsilon / 2^k$를 만족하면서 
각 $E_k$를 감싼다고 해보자. 그러면 countably subadditive 때문에 (i) 증명 완료!

### (i) => (ii)

알다시피 $G_{\delta}$는 intersection of countable collection of open sets.
이번엔 $E$를 감싸는 $\mathcal{O}_k^\prime$들을 떠올려서, $m^*(\mathcal{O}_k^\prime \sim E) < 1/k$라고 해서
그럼 "$G = \bigcap ^\infty _{k=1} \mathcal{O}_k^\prime$"라고 하면, $G$는 $E$를 감싸고 (ii)를 만족해.

### (ii) => measurable

(ii)가 성립한다면, $\sigma$-algebra에 의해 $E = G \cap [G \sim E]^C$는 measurable하겠지

아무튼 겉껍질의 존재성이 드러났다!!

나머지 (iii)이랑 (iv)는 $\sigma$-algebra에 의해
complement도 measurable하다고 할 수 있으니까 드모르간으로 대충 증명될거.
closed set $F$에 complement를 붙인 것을 open set $O$라고 한다면,
$E \sim F = E \cap F^C = E \cap O = O \sim E^C$
$E^C$를 감싸는 $O=F^C$가 완성되고, (i)의 조건에 부합하게 되지.

### P17. 겉껍질과 속껍질을 $\epsilon$만큼 좁히기

>Show that a set $E$ is measurable if and only if for each $\epsilon > 0$, there is a closed set $\mathcal{F}$ and open set $\mathcal{O}$ for which $\mathcal{F} \subseteq E \subseteq \mathcal{O}$ and $m^*(\mathcal{O} \sim \mathcal{F}) < \epsilon$

이렇게 $m^*(\mathcal{O} \sim \mathcal{F}) < \epsilon$로 **$\mathcal{O}$와 $\mathcal{F}$를 좁히는 식으로 해석할 수도 있겠지**
(=>)는 우리의 본 증명에서 $\epsilon$을 $\epsilon / 2$로 바꾸고 
$m^*(\mathcal{O} \sim E)$과 $m^*(E \sim \mathcal{F})$을 excision property로 합치면 그만.
(<=)는 (ii)로 넘어가서 $m^*(\mathcal{O} \sim F) = 0$니까 $\mathcal{O} \sim E$가 더 조그마하니 원래의 증명대로 유도 가능

### P18. 아예 겉껍질, 속껍질과 measure가 같다고 할 수도 있다

>Let $E$ have finite outer measure. Show that there is an $F_\sigma$ set $F$ and a $G_\delta$ set $G$ s.t
>$$
F \subseteq E \subseteq G \mathsf{\; and \;} m^*(F) = m^*(E) = m^*(G)$$

$E$의 measurability에서 $m^*(G \sim E) = 0$, $m^*(E \sim F) = 0$ 유도하고,
excision property로 해결.
**measureable만 하다면 껍질들과 아예 outer measure가 같단 결론이 나오네!!**

## 1-3. 껍질 하나로 충분할 수도 있다

>Let $E$ be a measurable set of finite outer measure.
>Then for each $\epsilon > 0$, there is a finite disjoint collection of open intervals $\{I_k\}^n_{k=1}$ 
>for which if $\mathcal{O} = \bigcup ^n _{k=1} I_k$, then
>$$
m^*(E \sim \mathcal{O}) + m^*(\mathcal{O} \sim E) < \epsilon$$

이젠 **아예 껍질 하나를 만들어서 $E$와 nearly equal한걸 만드네!**

$$
E \subseteq \mathcal{U} \mathsf{\; and \;} m^*(\mathcal{U} \sim E) < \epsilon /2$$
방금 1-2의 theorem으로 인해 open set $\mathcal{U}$를 만들 수 있고,
이런 조건이니 $\mathcal{U}$도 "finite" outer measure라고 할 수 있겠네

모든 open set of real number는 open interval의 countable collection이라고 할 수 있으니,
$\mathcal{U}$도 union of countable disjoint open intervals $\{I_k\}^\infty _{k=1}$이라고 할 수 있어

그럼 유한한 $n$에 대해 이런 일이 가능하지
$$
\sum ^n _{k=1} l(I_k) = m^*(\bigcup ^n _{k=1} I_k) \leq m^*(\mathcal{U}) < \infty$$
여기서 $n$에 대해 독립적이니 $\sum ^\infty _{k=1} l(I_k) < \infty$이고, 급수가 수렴하니 $\sum ^\infty _{k=n+1} l(I_k) < \epsilon / 2$
이 두가지 방식을 합칠줄은 몰랐네! <u>독립적이라 $\infty$ 보내버리고</u>, <u>수렴 조건 이용하기</u>.

"$\mathcal{O} = \bigcup ^{n} _{k=1} I_k$"라고 한다면? $\mathcal{O} \sim E \subseteq \mathcal{U} \sim E$니까 정의에 의해,
$m^*(\mathcal{O} \sim E) \leq m^*(\mathcal{U} \sim E) < \epsilon / 2$
$E \subseteq \mathcal{U}$이고, $E \sim \mathcal{O} \subseteq \mathcal{U} \sim \mathcal{O} = \bigcup _{k=n+1}^\infty I_k$니까 밝혀낸 것에 의해,
$m^*(E \sim \mathcal{O}) \leq \sum _{k=n+1} ^\infty l(I_k) < \epsilon / 2$

기가 막히게 $m^*(\mathcal{O} \sim E) + m^*(E \sim \mathcal{O}) < \epsilon$을 만들었다!!

![](https://velog.velcdn.com/images/7kong/post/74bdc08e-d66f-4de9-8caa-1f80497e3586/image.jpg)
**$\mathcal{O}$와 $E$의 차이를 빼고 남은 건더기들**이 $\epsilon / 2$를.
**끝까지 안 감쌌기에 남은 $I_k$들**에서 $\epsilon /2$를 만들어냈다.
끝까지 안 감싼단 생각 하기는 진짜 힘들 것 같긴 하다;;

# 1-4. 애초에 그렇게 안전하지 않은 excision

너무 자연스럽게 흘러와서 까먹었을 수 있지만,

For any $\epsilon > 0$, there is an open set $\mathcal{O}$ s.t $E \subseteq \mathcal{O}$ and $m^*(\mathcal{O}) < m^*(E) + \epsilon$
이건 $E$가 measurable하든 안 하든 성립하거든??

근데 여기서 $m^*(\mathcal{O}) - m^*(E) < \epsilon$이 되니까 $m^*(\mathcal{O} \sim E) < \epsilon$이 된다??
이건 완전히 다른 이야기야. **excision property는 $E$가 measurable이여야 성립해!**

## P19. measurable하지 않을 때 excision property는 깨진다

>Let $E$ have finite outer measure. Show that if $E$ is not measurable,
>then there is an open set $\mathcal{O}$ containing $E$ that has finite outer measure and for which
>$$
m^*(\mathcal{O} \sim E) > m^*(\mathcal{O}) - m^*(E)$$

not measurable 하니까 
open set $\mathcal{O}$가 $E$를 포함한다면,
$m^*(\mathcal{O}) < m^*(\mathcal{O} \cap E) + m^*(\mathcal{O} \cap E^C) = m^*(E) + m^*(\mathcal{O} \sim E)$
아니면 $m^*(\mathcal{O}) > m^*(E) + m^*(\mathcal{O} \sim E)$

근데 countably subadditive 때문에 $m^*(\mathcal{O}) \leq m^*(\mathcal{O} \cap E) + m^*(\mathcal{O} \cap E^C)$니까
$$
m^*(\mathcal{O}) < m^*(E) + m^*(\mathcal{O} \sim E) \Rightarrow m^*(\mathcal{O}) - m^*(E) < m^*(\mathcal{O} \sim E)$$

이것도 뭔가 직관을 깨부술 이야기가 나오겠다는 초석을 깔아둔거겠지.
또다시 measurable이 어디에나 먹히는 안전한 애인지의 문제가 대두되었어!

# 🐚2. measure 자체의 속껍질 겉껍질?

## 2-1. 두 껍질로 만든 measure

겉껍질의 inf로 만든 measure와
속껍질의 sup으로 만든 measure가 있다면 어떨까?

### P22. 겉껍질로 만든 measure

>For any set $A$, define $m^{**}(A) \in [0, \; \infty]$ by
>$$
m^{**}(A) = \mathsf{inf} \{m^*(\mathcal{O}) | \mathcal{O} \supseteq A, \; \mathcal{O} \; \mathsf{open}\}$$
>How is this set function $m^{**}$ related to outer measure $m^*$?

$m^*(\mathcal{O})$의 정의에 의해 대충 조건 맞는 $\{I_k\}$가 존재해서, $\bigcup _{k=1}^\infty I_k \supseteq \mathcal{O}$
$\mathcal{O} \supseteq A$니까 $\sum _{k=1}^\infty I_k \geq m^*(\mathcal{O}) \geq m^*(A)$
$m^{**}(A)$가 $\mathsf{inf}$니까 $m^{**}(A) \geq m^*(A)$

$m^{**}(A)$ 정의에 의해
$\sum _{k=1}^\infty I_k \geq m^*(\mathcal{O}) \geq m^{**}(A)$
$m^{*}(A)$가 $\sum _{k=1}^\infty I_k$의 $\mathsf{inf}$니까 $m^{*}(A) \geq m^{**}(A)$

그러니 $m^{*}(A) = m^{**}(A)$. **둘은 같은 measure다!**

### P23. 속껍질로 만든 measure

>For any set $A$, define $m^{***}(A) \in [0, \; \infty]$ by
>$$
m^{***}(A) = \mathsf{sup} \{m^*(F) | F \subseteq A, \; F \; \mathsf{closed}\}$$
>How is this set function $m^{***}$ related to outer measure $m^*$?

일단 $m^{*}(A) \geq m^{***}(A)$는 쉽거든
$m^{***}(A) \geq m^*(F)$, $A \supseteq F$니까 $m^*(A) \geq m^*(F)$여서 $\mathsf{sup}$ 사용하면 되니까.

근데 아무래도 $m^{*}(A) \leq m^{***}(A)$는 **$A$가 measurable이여야 되는 모양이야**
$A$가 measurable하면 위의 1-2의 정리에 의해 $m^*(A \sim F) < \epsilon$인 $F$ 존재.
$$
m^*(A) < m^*(F) + \epsilon \Rightarrow m^*(A) - \epsilon < m^*(F)$$
$m^{***}(A)$는 이런 $m^*(F)$들의 $\mathsf{sup}$이니까 $\epsilon$ 임의인거 이용해서 $m^{*}(A) \leq m^{***}(A)$ 도출 가능
그렇게 해서 $A$가 measurable하다면, $m^*(A) = m^{***}(A)$

$m^*(A) \neq m^{***}(A)$인 경우가 있는가? 이건 아무래도 뒤의 얘기가 나와야 알 수 있을거야.

## 2-2. 우리가 생각할 수 있는 진짜 measure

유독 우리가 지금까지 배우던 measure 이름을 "outer" measure라고 했단 말야?
그리고 지금 배우고 있는건 measurable set의 겉껍질과 속껍질인 
outer, inner approximation.
그리고 $A$가 measurable하다면, $m^{*}(A) = m^{**}(A) = m^{***}(A)$야

처음에 말했다시피 outer measure의 정의를 보면
바깥에서 정하는 크기와 안에서 정하는 크기가 합쳐져 
우리가 원하는 크기가 정해질 것만 같았어.

우린 measurable set에서 outer measure라는 set fucntion을 통해
크기를 정의하고 싶으니까, 
outer measure가 바깥과 안에서 크기를 정의하듯이,
measurable set이란 것도 겉껍질과 속껍질로 좁힐 수 있단걸 드러내야 하기에
이번 챕터가 있다고 생각했지.

근데 사실 **measure 자체도** outer measure의 외부와 내부 성질 이상으로
**다른 겉껍질 속껍질 정의가 필요하다면?**
outer measure $m^*(A)$는 보면 결국 $\mathsf{inf}$로만 정의한 애야. 그리고 $m^{**}(A)$랑 같았지.
measurable하지 않다면 $m^{***}(A)$랑 같지 않았어.
**정말로 measure가 정의되려면 $\mathsf{sup}$을 담당하는 inner measure? 같은 애가 필요하지 않을까?**
**outer measure와 inner measure가 같아져야 하는 일련의 일이 필요하지 않을까??**
앞으로 진짜 measure가 나올 때 어떤 정의가 나올지 기대해보자.

{제미나이}
여담으로 이 논리는 AI의 Duality 이론과 아주 많이 닮아있대.
최적화 문제에서 원래 문제인 Primal의 상한과 보조 문제인 Dual의 하한을 좁혀서
Duality Gap을 0으로 만드는 방식으로 한다네.
근데 얘기 들어보면 실해석학과 대놓고 관련이 있진 않아 보이고,
논리만 관련 있을 것 같은데,
나중에 저 Duality가 어떤건지 좀 알아보고, 이 겉껍질, 속껍질과 연결지어봐도 재밌을 것 같아.


## 3. 반례: $m^*(A) \neq m^{**}(A)$인 경우 (비가측 집합)

가장 대표적인 반례는 **비탈리 집합(Vitali Set)** 또는 가측 집합이 아닌 임의의 집합을 통해 생각할 수 있습니다. 하지만 더 직관적인 이해를 돕기 위해 다음과 같은 성질을 가진 집합 $V \subset [0, 1]$를 가정해 보겠습니다.

### 반례의 구성

어떤 집합 $V$가 **"내측도(Inner measure)가 0이고, 외측도(Outer measure)가 1"**인 경우를 생각해 봅시다. (예: $m^*(V) = 1$이고 $m^*([0, 1] \setminus V) = 1$인 비가측 집합)

1. $V$ 내부에 포함된 임의의 닫힌 집합 $F$를 잡습니다.
    
2. 만약 $V$가 "매우 성긴" 비가측 집합이라서, 그 안에 포함될 수 있는 닫힌 집합이 오직 가산 집합(countable set)이나 측도가 0인 집합들뿐이라면 어떻게 될까요?
    
3. 그러면 모든 $F \subseteq V$에 대해 $m^*(F) = 0$이 됩니다.
    
4. 이 경우 $m^{***}(V) = \sup \{0\} = 0$ 이지만, $m^*(V) = 1$이 되어 두 값은 같지 않습니다.
    

### 결론

$m^{***}(A) = m^*(A)$라는 조건은 사실상 **$A$가 가측(measurable)이라는 조건과 동치**입니다. (단, $m^*(A) < \infty$일 때). 따라서 비가측 집합을 찾기만 하면 그것이 곧 반례가 됩니다.

혹시 비탈리 집합의 구체적인 구성이나, $m^*(A) = \infty$인 경우의 처리에 대해 더 자세한 설명이 필요하신가요?