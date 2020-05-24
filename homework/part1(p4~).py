#4페이지


#users 라는 리스트에 각각 딕셔너리를 생성, 각 딕셔너리에는 id와 name이라는 키가 있다.
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

#p5

# 딕셔너리 생성, friendship에는 각 id에 연결된 친구를, 리스트 안에 저장하려고 한다.
friendships = {user["id"]: [] for user in users } #리스트 컴프리헨셔 # 0:[],1:[]

print(friendships)#{0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

for i,j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

print(friendships)#0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4, 6, 7], 6: [5, 8], 7: [5, 8], 8: [6, 7, 9], 9: [8]

def number_of_friends(user):#user에는 
    return len(friendships[user["id"]])#각 아이디를 통해서 몇 명의 친구가 있는지 반환하는 함수

# print(number_of_friends(users[0]))
# number_of_friends, users에 있는 딕셔너리를 넣으면 친구의 숫자 호출

#p6
# 딕셔너리 순서대로 호출해서 몇 명의 쌍이 있는지 확인
total_connections =sum(number_of_friends(user) for user in users) 

print(total_connections)#24

num_users=len(users)

avg_connections = total_connections/num_users #24/10 == 2.4

num_friends_by_id = [(user['id'],number_of_friends(user))for user in users]

print(num_friends_by_id)#[(0, 2), (1, 3), (2, 3), (3, 3), (4, 2), (5, 3), (6, 2), (7, 2), (8, 3), (9, 1)]

num_friends_by_id.sort(key=lambda id_and_friends : id_and_friends[1],reverse=True) #number_of_friends(user)의 내림차순 순으로 정렬

print(num_friends_by_id)#[(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]


#p7 - 친구의 친구 소개시켜주기
def foaf_bad(user): #friend of a friend
    return [foaf_id for friend_id in friendships[user["id"]] for foaf_id in friendships[friend_id]]

print(list(foaf_bad(user) for user in users))# [[0, 2, 3, 0, 1, 3], [1, 2, 0, 1, 3, 1, 2, 4], [1, 2, 0, 2, 3, 1, 2, 4], [0, 2, 3, 0, 1, 3, 3, 5], [1, 2, 4, 4, 6, 7], 
#[3, 5, 5, 8, 5, 8], [4, 6, 7, 6, 7, 9], [4, 6, 7, 6, 7, 9], [5, 8, 5, 8, 8], [6, 7, 9]]

#p8

from collections import Counter

def friends_of_friends(user):
    user_id=user["id"]
    return Counter(foaf_id for friends_id in friendships[user_id] for foaf_id in friendships[friends_id] if foaf_id != user_id and foaf_id not in friendships[user_id])

print(friends_of_friends(users[3]))