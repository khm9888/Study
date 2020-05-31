#PS4Q7
#Write a program that computes the value of a+aa+aaa+aaaa
#with a given digit as the value of a.
#Example: Suppose the following input is given:
#5
#Then, the output should be: 6170 
#(5 + 55 + 555 + 5555)
#(the main part of the code should be inside a function.
#So take input from the user, and then pass it to a function
#called calculateAValue().)





#ALGORITHM
#take input digit convert to int
#def function....calc value , takes one int value only
    #set int to a
    #return this expression
#call function..pass int...print return value



















#code
userInt = int(input("Type an integer"))

def calculateAValue(a):
    a = a
    aa = int(str(a)*2)
    aaa =int(str(a)*3)
    aaaa = int(str(a)*4)
    return a+aa+aaa+aaaa

print(calculateAValue(userInt))
