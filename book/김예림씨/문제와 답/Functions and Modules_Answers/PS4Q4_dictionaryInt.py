#PS4Q4
#With a given integral number n,
#write a program to generate a dictionary
#that contains (i, i*i) such that is an integral
#number between 1 and n (both included).
#and then the program should print the dictionary.
#Suppose the following input is supplied to the program:
#7
#Then, the output should be:
#{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49}
#(the main part of the code should be inside a function.
#So take input from the user, and then pass it to a function
#called generateSquares(). )




#ALGORITHM
#take int input...
#def function...takes one interger
    #create empty dictionary
    #start a loop...iterate over range of that int
        #add k-v pair to dict ...k is the iterator+1...and value is (k+1)**2
    #return this final dictionary
#call function...pass the interger and
#print the dictionary you got from the function




















#code
userInt = int(input("Type an integer"))

def generateSquares(int1):
    dict1 = {}
    for i in range(int1):
        dict1[i+1] = (i+1)**2
    return dict1

print(generateSquares(userInt))

