'''
x = input("숫자를 입력하세요  :")
sum = 0
for i in x:
    sum +=int(i)
print(sum)
'''
'''
x = int(input("숫자를 입력하세요  :"))
sum = 0
while x>=10 :
    sum+=x%10
    x = int(x/10)
    
sum +=x
print(sum)
'''

list = [1,2,3,4,5]

print(list[::-1])

for x in list[::-1]:
    print(x)
