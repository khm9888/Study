#PS4Q6
#Write a program which takes 2 digits (on same line),
#X,Y as input and generates a 2-dimensional array.
#The element value in the i-th row and j-th column of the array should be i*j.
#Note: i=0,1.., X-1;  j=0,1,..., ­Y-1.
#Example.
#Suppose the following inputs are given to the program:
#3,5
#Then, the output of the program should be:
#[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]
#(the main part of the code should be inside a function.
#So take input from the user, and then pass it to a function
#called generateArray(). )





#ALGORITHM
#take input from user two numbers separated by ','...so split on that.

#def function....gen array ...pass two integers to it
    #create matrix list
    #start iteration over row...j...y:
        #rowlist...empty list
        #start iteration over column ...i...x:
            #append values to rowlist..
        #append this rowlist completely to matrix list
    #return the final matrix list

#call function ....pass the two values in the list (split from user input)
#print the value returned















#code
rowsAndColumns = input("Type rows and columns needed in i,j format").split(",")

def generateArray(i,j):
    #i--rows
    #j--columns
    matrixList = []
    for everyi in range(i):
        rowList = []
        for everyj in range(j):
            rowList.append(everyi*everyj)
        matrixList.append(rowList)
    return matrixList

print(generateArray(int(rowsAndColumns[0]),int(rowsAndColumns[1])))
            
