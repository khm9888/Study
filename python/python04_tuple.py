"""자료형"""
"""
2. 튜플
소괄호로 입력
리스트와 비슷하지만 삽입, 삭제, 수정이 안됨
"""

a = (1, 2, 3)
b = 1, 2, 3
print(type(a))
print(type(b))

# a.remove(2)
# print(a)

print(a + b)
print(a * 3)

# print(a - 3)    # TypeError: unsupported operand type(s) for -: 'tuple' and 'int'