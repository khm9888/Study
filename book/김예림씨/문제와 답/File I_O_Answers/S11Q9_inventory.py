#S11Q9
#Download file: https://drive.google.com/file/d/1tvxpyV1Iva0a0BNnzSbEVRUGVWK4_6x9/view?usp=sharing 
#This is the inventory of the supermarket where Ken works.
#His boss comes to him and gives him the following information.
#mangoCount = +75 pieces #increased by 75
#orangeCount = ? #still unknown
#bananasCount = 20 pieces 
#pearsCount = 100 pieces #final count

#Take this information into account and update the warehouseInventory.txt you downloaded. 
#Final contents of warehouseInventory.txt file
#Inventory:
#Apples: 30 pieces
#Mangoes: 100 pieces
#Oranges: ?
#Bananas: 20 pieces
#Carrots: 27 pieces
#Pears: 100 pieces








#ALGORITHM
#DON'T AUTOMATE THIS EXAMPLE!
#JUST DO IT THE SIMPLEST WAY POSSIBLE

#open file...read lines ...each line is list element
#close the file
#make a collection of update actions required.
#make a function that can find which index the element that needs to be
#updated is at.
#call function for mangoes
#get the index and update...
#similarly for bananas and pears

#open file again..this time to write...
#write the entire new list of lines..
#close the file











#CODE
f = open("warehouseInventory.txt",'r')
#f.seek(0)
list1= f.readlines()
#print(list1)
f.close()

changeActions = {"Mangoes":(75,"add"),"Bananas":(20,"final"), "Pears":(100,"final") }

def findFruitIndex(str1,list1):
    i1 = 0
    for everyString in list1:
        if str1 in everyString:
            i1 = list1.index(everyString)
    return i1
    
#find mangoes
index1 = findFruitIndex("Mangoes:",list1)
#update mangoes
list1[index1] = "Mangoes: 100 pieces\n"
#find bananas
index1 = findFruitIndex("Bananas:",list1)
#update bananas
list1[index1] = "Bananas: 20 pieces\n"
#find pears
index1 = findFruitIndex("Pears:",list1)
#update pears
list1[index1] = "Pears: 100 pieces\n"

f = open("warehouseInventory.txt",'w')
f.write("".join(list1))
f.close()
                


