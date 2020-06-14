# list_1st = input('정수를 여러 개 입력하시오 : ').split()
# list_2nd = [int(i) for i in list_1st]

# def mean_of_n(list_values):
#     total = 0
#     for i in list_values:
#         total = total + i
#     average = total / len(list_values)
#     return average

# print(mean_of_n(list_2nd))

# print(sum(list_2nd)/len(list_2nd))

# import time

# start = time.time()

# end = time.time()
# input()
# if end-start>=3:
#     print("timeover")
# else:
#     print("pass")

# values = input("숫자 3개 입력하세요").split()
# values = [int(i) for i in values]

# minimum = min(values)
# # print(minimum)

# 공약수=1 #영어모름
# for i in range(minimum,1,-1):
#     if values[0]%i ==0 and values[1]%i ==0 and values[2]%i ==0:
#         공약수 = i
#         break
# print(공약수)

# # alist=["1"]
# # alist.insert(1,"2")

# # print(alist)

l1 = [1,2,3,4]
l2 = [3,4,5]
l3=l1+l2
print(l3)