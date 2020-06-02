#PS4Q2
#Take a random string from the user.
#Calculate and print separately, the number of letters,
#spaces and digits in that string.
#To do this create a separate module file requiredFunctions.py.
#In this module file, there should be three separate functions
#calcSpaces(), calcDigits() and calcLetters().
#Call this module into your main code. 
#Example.
#Input - Hi, I am Aditya. I’m an elf and my real age is 300 years.
#Output -
#Number of letters - 37
#Number of digits - 3
#Number of spaces - 13







#ALGORITHM
#--main code
#import module
#input string from user
#call space calc function
#print result
#call digit calc function
#print result
#call letters calc function
#print result

#--module
#def space calc function...takes one string
    #set a counter to 0
    #iterate over string
        #check if space
            #if yes, then increment counter
    #return final count

#def digits calc function...takes one string
    #set a counter to 0
    #iterate over string
        #check if number...isdigit
            #if yes, then increment counter
    #return final count

#def letters calc function...takes one string
    #set a counter to 0
    #iterate over string
        #check if letter...isalpha
            #if yes, then increment counter
    #return final count



#code
import requiredFunctions
userString = input("Type string")

num = requiredFunctions.calcSpaces(userString)
print("Number of spaces: ",num)

num = requiredFunctions.calcDigits(userString)
print("Number of digits: ",num)

num = requiredFunctions.calcLetters(userString)
print("Number of letters: ",num)

















