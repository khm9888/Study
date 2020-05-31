#S11 q1
#Write a program that creates a text file NameOfUser .txt,
#takes the First Name and Last Name from the user and writes it into the file.
#Input:
#Your full name please: Ken Nicholson
#Output: 
#Contents of Text file.
#Name of person is: Ken Nicholson.



#ALGORITHM

#open a text file

#take string input first name
#take string input last name

#write both to file .write()
#clode file


















#CODE

f = open("NameOfUser.txt",'w')

firstName = input("Type your first name")
lastName = input("Type your last name")

f.write(firstName + " " + lastName)

f.close()




