kind = int(input())
values =[]
for i in range(kind):
    x,y = map(int,input().split())
    y=y%4
    if y==0:
        y=4
    input_data = (x**(y))%10
    if input_data==0:
        input_data = 10
    values.append(input_data)

for i in values:
    print(i)