"""자료형"""
"""
1. 리스트
대괄호로 입력
리스트 내부에는 여러가지 자료형이 올 수 있음
"""

a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 'a', 'b', '']    # 숫자형과 문자형
print(b)

print(a[0] + a[3])    # 리스트 a의 0번째 인덱스 : 1, 3번째 인덱스 : 4
# print(b[0] + b[3])    # TypeError, 자료형이 맞지 않음

print(type(a))    # type() 인자로 들어간 자료형의 타입을 반환
print(a[-2])
print(a[1:3])    # 첫번째 인덱스 ~ 두번째 인덱스(3 - 1), 끝에 지정해준 숫자보다 하나 적은 인덱스까지를 반환

a = [1, 2, 3, ['a', 'b', 'c']]    # 리스트 a는 리스트 안에 리스트가 원소(인덱스)로 들어간 형태
print(a[1])    # res = 2
print(a[-1])    # res = ['a', 'b', 'c']
print(a[-1][1])    # 원소 리스트의 원소를 뽑는 방법 ; res = b

# 1-2. 리스트 슬라이싱
a = [1, 2, 3, 4, 5,]
print(a[:2])    # 앞에 아무것도 명시되어 있지 않으면, '처음'부터를 의미함, res = [1, 2]

# 1-3. 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)    # 두 개 이상의 리스트를 + 연산하면 각 리스트의 원소들이 하나로 나열된, 하나의 리스트가 반환됨
                # res = [1, 2, 3, 4, 5, 6]

c = [7, 8, 9, 10]
print(a + c)

print(a * 3)

# print(a[2] + 'hi')
print(str(a[2]) + 'hi')

f = '5'
# 1. a[2]를 str()로 형변환
print(str(a[2]) + f)

# 2. f를 int()로 형변환
print(a[2] + int(f))


# 리스트 관련 함수
a = [1, 2, 3]
# 1-1. append()
a.append(4)
print(a)
# a = a.append(5)
# print(a)          # Error : None이 return됨.

# 1-2. sort()
a.sort()
print(a)

# 1-3. reverse()
a.reverse()
print(a)    # [4, 3, 2, 1]

# 1-4. index()
print(a.index(3))   # == a[3]
print(a.index(1))   # == a[1]

# 1-5. insert()
a.insert(0, 7.1)    # insert(삽입할 자리, 삽입할 숫자)
print(a)    # [7.1, 4, 3, 2, 1]

a.insert(3, 3)
print(a)    # [7.1, 4, 3, 3, 2, 1]

# 1-6. remove()     # remove() 지정한 원소를 삭제함
a.remove(7.1)
print(a)    # [4, 3, 3, 2, 1]

a.remove(3)         # 지정한 원소가 하나 이상일 경우 앞에 위치한 원소부터 삭제
print(a)    # [4, 3, 2, 1]