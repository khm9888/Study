#4페이지


users =[
    {"id":  0, "name":"Hero"},
    {"id":  1, "name":"Dunn"},
    {"id":  2, "name":"Sue"},
    {"id":  3, "name":"Chi"},
    {"id":  4, "name":"Thor"},
    {"id":  5, "name":"Clive"},
    {"id":  6, "name":"Hicks"},
    {"id":  7, "name":"Devin"},
    {"id":  8, "name":"Kate"},
    {"id":  9, "name":"Klein"}
]


# print(list(users[0].keys()))
# print(type(user[0][""]))


#친구관계 연결된 리스트
friendship_pairs=[(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
# print(len(friendship_pairs)) #12

friendships = {user["id"]: [] for user in users } #리스트 컴프리헨셔 # 0:[],1:[]

print(friendships)#{0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

for i,j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

print(friendships)#0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4, 6, 7], 6: [5, 8], 7: [5, 8], 8: [6, 7, 9], 9: [8]

def number_of_friends(user):
    return len(friendships[user["id"]])

print(number_of_friends(users[0]))# number_of_friends, users에 있는 딕셔너리를 넣으면 친구의 숫자 호출

total_connections =sum
