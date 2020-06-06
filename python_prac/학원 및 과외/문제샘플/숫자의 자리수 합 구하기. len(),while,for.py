#input을 통해서 임의의 숫자를 입력받고, 각 자리 수의 숫자의 합을 구하라.
#len() #for if%10==0
#code1
'''
x = input("숫자를 입력하세요  :")
sum = 0
for i in x:#x=12
    sum+=int(i)           #i="1","2"
    sum=sum+int(i)
print(sum)

#code2
for j in range(len(x)):#len(x)=2 , j=0,1
    sum+=int(x[j])#x[0]="1",x[1]="2" 
 '''   
#code3
x = int(input("숫자를 입력하세요  :"))#int 형으로 받는 경우
sum_user = 0
while x>=10 : #12
    sum_user+=x%10 #1, sum=sum+x%10
    x = int(x/10)

sum_user +=x
print(sum_user)




