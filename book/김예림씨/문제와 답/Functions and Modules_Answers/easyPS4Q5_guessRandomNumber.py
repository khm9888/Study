#PS4Q5
#Number Guessing Game: 
#Write a program that accepts a number from the
#user (any integer between 1 and 10).
#Your program should also generate a random number.
#Between 1 and 10. If the number the user entered is
#the same as the one your program generated the user wins.
#If not, the user gets three tries, and if unsuccessful, the user loses. 
#Hint: check out the randint() function in Python3
#Python documentation: https://docs.python.org/3.1/library/random.html




#ALGORITHM
#import the random module
#start a counter, set it to 0
#start game...start an infinite loop...everything will be inside it
    #ask user for a number..input...int. between 1 and 10
    #generate random number between 1 and 10....randint()
    #compare both numbers..if
        #if yes then say you won and break out
    #if not...else...raise counter value by 1
    #check if counter value has reached 3
        #if yes...say you lost...and break
    #else go ahead
        

















#code
import random
print("Welcome to the guessing game")
attempts = 1
while True:
    userInt = input("Type a number")
    sysInt = random.randint(1,10)
    if attempts == 3:
        print("You lost!")
        break
    else:
        if sysInt == userInt:
            print("You won!")
            break
        else:
            print("Try again")
            attempts+=1
    
    
    
