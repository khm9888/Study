#S11Q10
#Download file 1: 
#https://drive.google.com/file/d/1P_8RK3J-RaE_pK-gfvDebqzuHtZOuO1P/view?usp=sharing 

#Download file 2:
#https://drive.google.com/a/student.utwente.nl/file/d/16BQqQGuyRY_1Vv4HnvApyBz5AIJIypWv/view?usp=sharing 

#You are given two text files. Read both files in the same Python program.
#Find out which file is in Finnish and which is in Dutch. 

#Note: The file with the higher number of ‘e’ type characters
#is Dutch and the file with the higher number of ‘a’ type characters is Finnish.

#Also check out the Extended ASCII characters section!
#https://theasciicode.com.ar/ 
#Hint: There are other types of ‘e’ and ‘a’ too!




#ALGORITHM
#define a function....this function will find count of a and e in any string passed
#to it
    #start a counte and counta variables...set to 0
    #loop over each character of f.read()..
        #if counta is any of ...ascii numbers (65,97,131-134,142-143,160,181-183,198,199)
            #increment counta
        #if counte is any of ...ascii numbers (69,101,130,136-138,144,210-212)
            #increment counte

#open file 1 to read
#.read everything
#call function...
#print results of first file...

#<do exact same for second file>

#compare and print the result...final judgement.















#CODE


def countAandE(text):
    ca = ce = 0
    for everyChar in text:
        if ord(everyChar) in [65,97,131,132,133,134,142,143,160,181,182,183,198,199]:
            #65,97,131-134,142-143,160,181-183,198,199)
            ca+=1
        if ord(everyChar) in [69,101,130,136,137,138,144,210,211,212]:
            #(69,101,130,136-138,144,210-212)
            ce+=1
    return ca,ce

#file1
f = open("languageText1.txt",'r')
fileText = f.read()
counta1,counte1 = countAandE(fileText)
print("File 1 Number of a characters:",counta1)
print("File 1 Number of e characters:",counte1)
f.close()

#file2
r = open("languageText2.txt",'r')
fileText = r.read()
counta2,counte2 = countAandE(fileText)
print("File 2 Number of a characters:",counta2)
print("File 2 Number of e characters:",counte2)
r.close()

#final judgement

if counta1 > counta2 and counte1 < counte2:
    print("It's clear. The first file is Finnish. And the second's Dutch")
elif counta2 > counta1 and counte1 > counte2:
    print("The second file is Finnish. The first one's Dutch")
else:
    print("Making a judgement is inconclusive. Need more data or longer files")
    

    
