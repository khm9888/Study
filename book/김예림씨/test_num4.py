
#4번  - clear
''' 
Medium 4
Write a program that takes two lists of integer numbers and checks if they can be combined to form a ‘consecutive’ list.
If yes then print True, else print False.
Example1:
Input: Type numbers for List1: 2,6,5,3
Type numbers for List2 = 1,4,7
Output: True
(logic: complete sequence possible: 1,2,3,4,5,6,7)
Example2:
Input: Type numbers for List1 = 1,3,5,6
Type numbers for List2 = 2
Output: False
(logic: 4 is missing)
'''

#리스트 확인,map 확인
x = list(map(int,input("Type numbers for List1 =").split(",")))
y = list(map(int,input("Type numbers for List2 =").split(",")))

values = x+y
values.sort()

print(values)

chk=True
for i,j in enumerate(values):
    if i!=0:
        if j!=values[i-1]+1:
            # print(j!=values[i-1])
            print(j)
            print(values[i])
            chk=False   
print(chk)



