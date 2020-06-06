#1 - clear 파일입출력 부분 
'''
Download this file: https://drive.google.com/file/d/19GgfoLcQXbMBMZ45SBeUBSvOvOHJXUz0/view?usp=sharing
Find and print the number of times the phrase “abcd” appears in this file.
Output: The number of times abcd appears in this file is: (your answer here)
'''
#파일 입출력 문제

# f = open("D:\\Study/Study/book/김예림씨/randomLetters.txt","r")
# lines = f.readlines()
# for line in lines:
#     # text = "aghirgheabcdhithabcgghfabcdaoirytsacdfgabcdyy"
#     cnt=0
#     for i,j in enumerate(line):
#         if line[i:i+4]=="abcd":
#             cnt+=1
    
# print("The number of times abcd appears in this file is:"+str(cnt))
# f.close()

#2번 - clear
'''
Download this file: https://drive.google.com/file/d/1aQVOyon9JJ7GZved8ZLfP-l9w7tapLmv/view?usp=sharing
Replace the marks Florentina got with 78.
You should make the replacement in the same file.
'''
#파일 입력 및 수정
'''
No Name Marks
1 Adi 45
2 Krupa 56
3 Florentina 14
4 Erik 96
# '''
# f = open("D://Study/Study/book/김예림씨/\studentMarks.txt","r")
# lines_list=[]
# lines = f.readlines()
# for line in lines:
#     cnt=0
#     for i,j in enumerate(line):
#         if line[i:i+len("Florentina 14")]=="Florentina 14":
#             line=line[:i]+"Florentina 79\n"
#             break

#     lines_list.append(line)
# f.close()
# f2 = open("D://Study/Study/book/김예림씨/\studentMarks2.txt","w")
# for line in lines_list:
#     f2.write(line)
# f2.close()
    
# # print("The number of times abcd appears in this file is:"+str(cnt))
# # f.close()


#3번-clear
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





#5번 -clear
'''
Medium 5
Write a program that has two functions: theLetters() and theNumbers().
The first function should print all the alphabets with both lowercase and uppercase letters on each line.
The second function should output all the numbers from 0 to 9 together with their names on each line.
The user chooses which function runs by introducing one of these strings: “numbers” or “letters”.
Functions Structure:
none theLetters():
none theNumbers():
Example1:
Input: What would you like to learn? numbers
Output:
0 zero
1 one
2 two
3 three
4 four
5 five
6 six
7 seven
8 eight
9 nine
Example1:
Input: What would you like to learn? letters
Output:
a A
b B
c C
.
.
.
y Y
z Z

'''

# import string

# lowercase = string.ascii_lowercase # 소문자 abcdefghijklmnopqrstuvwxyz
# string.ascii_uppercase # 대문자 ABCDEFGHIJKLMNOPQRSTUVWXYZ
# string.ascii_letters #대소문자 모두 abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
# numbers = string.digits # 숫자 0123456789 

# def theNumbers():
#     numbers=[]
#     numbers.append("0 zero")
#     numbers.append("1 one")
#     numbers.append("2 two")
#     numbers.append("3 three")
#     numbers.append("4 four")
#     numbers.append("5 five")
#     numbers.append("6 six")
#     numbers.append("7 seven")
#     numbers.append("8 eight")
#     numbers.append("9 nine")
#     for i in numbers:
#         print(i)

# def theLetters():
#     lowercase = string.ascii_lowercase # 소문자 abcdefghijklmnopqrstuvwxyz
#     for i in lowercase:
#         print(i,i.upper())


# choice = input("What would you like to learn?")
# if choice =="numbers":
#     theNumbers()
# elif choice == "letters":
#     theLetters()

#6번 - clear
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