#1
'''
Download this file: https://drive.google.com/file/d/19GgfoLcQXbMBMZ45SBeUBSvOvOHJXUz0/view?usp=sharing
Find and print the number of times the phrase “abcd” appears in this file.
Output: The number of times abcd appears in this file is: (your answer here)
'''
# #파일 입출력 문제
# text = "aghirgheabcdhithabcgghfabcdaoirytsacdfgabcdyy"
# cnt=0
# for i,j in enumerate(text):
#     if text[i:i+4]=="abcd":
#         cnt+=1
        
# print("The number of times abcd appears in this file is:"+str(cnt))


#2번
'''
Download this file: https://drive.google.com/file/d/1aQVOyon9JJ7GZved8ZLfP-l9w7tapLmv/view?usp=sharing
Replace the marks Florentina got with 78.
You should make the replacement in the same file.
'''
#파일 입력 및 수정


#3번
'''
Medium 3
Write a program that takes an integer number from the user and prints the following sequence (with the lines being equal to the number given).
Example1:
Input: Give a number: 3
Output:
*
**
***
Example2:
Input: 5
Output:
*
**
***
****
*****
'''

# x = int(input("Give a number:"))
# for i in range(1,x+1):
#     print("*"*i)


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

# #리스트 확인,map 확인
# x = input("Type numbers for List1 =").split(",")
# y = input("Type numbers for List2 =").split(",")
# # x=list(x)
# # y=list(y)

# #확인하기 - sort, sorted
# # x=x.sort()
# # y=y.sort()

# print(x)
# print(y)

'''
Hard 1
Write a program that takes a positive integer from the user and calculates how many dots exist in a pentagonal shape around the center dot on the Nth
iteration.
In the image below you can see that the first iteration is only a single dot. On the second, there are 6 dots. On the third, there are 16 dots, and on the
fourth, there are 31 dots.
Return the number of dots that exist in the whole pentagon on the Nth iteration.
Note: You DON’T have to create the shape. You just have to take the integer input from the user and print the output number of dots.
The main part of your code should be inside a function findPentagonal()
Function Structure:
int findPentagonal(int):
'''

# def findPentagonal(n):
#     values = 0
#     for i in range(1,n+1):
#         if i==1:
#             values+=1
#         else:
#             values+=(i-1)*5
#     return values

# x=int(input())

# print(findPentagonal(x))