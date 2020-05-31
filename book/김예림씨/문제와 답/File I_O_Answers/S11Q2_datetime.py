#S11 q2
#Write a program that creates a file currentDateTime.txt
#and write the current Date and Time in it, in the given format.
#Output:
#Contents of Text file.
#Entry 1: Tue May 19 09:45:06 2020

#Ref: check the Getting Formatted Time section: 
#https://www.tutorialspoint.com/python3/python_date_time.htm 





#ALGORITHM
#import time module..
#open the file to write
#get the current date time using function from date time module
#write this to file
#close the file















#code
import time
f = open("currentDateTime.txt", 'w')

localtime = time.asctime( time.localtime(time.time()) )

f.write(localtime)

f.close()
