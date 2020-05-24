# # 4페이지


# #users 라는 리스트에 각각 딕셔너리를 생성, 각 딕셔너리에는 id와 name이라는 키가 있다.
# users =[
#     {"id":  0, "name":"Hero"},
#     {"id":  1, "name":"Dunn"},
#     {"id":  2, "name":"Sue"},
#     {"id":  3, "name":"Chi"},
#     {"id":  4, "name":"Thor"},
#     {"id":  5, "name":"Clive"},
#     {"id":  6, "name":"Hicks"},
#     {"id":  7, "name":"Devin"},
#     {"id":  8, "name":"Kate"},
#     {"id":  9, "name":"Klein"}
# ]

# # print(list(users[0].keys()))
# # print(type(user[0][""]))

# #친구관계 연결된 리스트
# friendship_pairs=[(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
# # print(len(friendship_pairs)) #12

# #p5

# # 딕셔너리 생성, friendship에는 각 id에 연결된 친구를, 리스트 안에 저장하려고 한다.
# friendships = {user["id"]: [] for user in users } #리스트 컴프리헨셔 # 0:[],1:[]

# print(friendships)#{0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

# for i,j in friendship_pairs:#친구연결관계를 통해서 각 친구들을 추가한다.
#     friendships[i].append(j)
#     friendships[j].append(i)

# print(friendships)#0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4, 6, 7], 6: [5, 8], 7: [5, 8], 8: [6, 7, 9], 9: [8]

# def number_of_friends(user):#user라는 딕셔너리를 넣는다. users에 있는 딕셔너리 
#     return len(friendships[user["id"]])#각 아이디를 통해서 몇 명의 친구가 있는지 반환하는 함수

# # print(number_of_friends(users[0]))
# # number_of_friends, users에 있는 딕셔너리를 넣으면 친구의 숫자 호출

# #p6
# # 딕셔너리 순서대로 호출해서 몇 명의 쌍이 있는지 확인
# total_connections =sum(number_of_friends(user) for user in users) 

# print(total_connections)#24

# num_users=len(users)

# avg_connections = total_connections/num_users #24/10 == 2.4

# num_friends_by_id = [(user['id'],number_of_friends(user))for user in users]

# print(num_friends_by_id)#[(0, 2), (1, 3), (2, 3), (3, 3), (4, 2), (5, 3), (6, 2), (7, 2), (8, 3), (9, 1)]

# num_friends_by_id.sort(key=lambda id_and_friends : id_and_friends[1],reverse=True) #number_of_friends(user)의 내림차순 순으로 정렬

# print(num_friends_by_id)#[(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]


# #p7 - 친구의 친구 소개시켜주기
# def foaf_bad(user): #friend of a friend
#     return [foaf_id for friend_id in friendships[user["id"]] for foaf_id in friendships[friend_id]]

# print(list(foaf_bad(user) for user in users))# [[0, 2, 3, 0, 1, 3], [1, 2, 0, 1, 3, 1, 2, 4], [1, 2, 0, 2, 3, 1, 2, 4], [0, 2, 3, 0, 1, 3, 3, 5], [1, 2, 4, 4, 6, 7], 
# #[3, 5, 5, 8, 5, 8], [4, 6, 7, 6, 7, 9], [4, 6, 7, 6, 7, 9], [5, 8, 5, 8, 8], [6, 7, 9]]

# #p8

# from collections import Counter

# def friends_of_friends(user):
#     user_id=user["id"]
#     return Counter(foaf_id for friends_id in friendships[user_id] for foaf_id in friendships[friends_id] if foaf_id != user_id and foaf_id not in friendships[user_id])

# print(friends_of_friends(users[3]))#3번의 친구는 1,2,4 번이다. 3번은 모르면서 1번이 알고 있는 친구는 0번, 3번은 모르면서 2번이 알고있는 친구는 0번
# #3번이 모르면서 4번이 알고 있는 친구는 5번이기에
# #Counter({0: 2, 5: 1})
# #와 같은 결과가 나온다.

# #friends_of_friends
# print(list(friends_of_friends(user) for user in users))
# #0번부터 해서 순서대로 카운트된다.

# #p8

# #각자의 user_id와, 관심사를 저장한 리스트(item:튜플)
# interests = [
#     (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
#     (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
#     (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
#     (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
#     (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
#     (3, "statistics"), (3, "regression"), (3, "probability"),
#     (4, "machine learning"), (4, "regression"), (4, "decision trees"),
#     (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
#     (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
#     (6, "probability"), (6, "mathematics"), (6, "theory"),
#     (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
#     (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
#     (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
#     (9, "Java"), (9, "MapReduce"), (9, "Big Data")
# ]

# #p9

# def data_scientists_who_like(target_interest):
#     return [user_id
#             for user_id, user_interest in interests
#             if user_interest == target_interest]
    
# print(data_scientists_who_like("regression"))#관심사를 넣고, 해당관심사를 가지고 있는 사람 출력

# from collections import defaultdict

# user_ids_by_interest = defaultdict(list)#모든 key에 대한 value의 기본값을 list가 있는 형태

# for user_id, interest in interests: #관심도 조사에 따른
#     user_ids_by_interest[interest].append(user_id)#각 관심도에 id 추가
    

# interests_by_user_id = defaultdict(list)

# for user_id, interest in interests:
#     interests_by_user_id[user_id].append(interest)#각 id에 관심도 추가

# print(f"interests_by_user_id[3]:{interests_by_user_id[3]}")
# print(f"user_ids_by_interest['R']:{user_ids_by_interest['R']}")

# #확인사항.
# '''
# #구현되지 않아서 확인해야함
# def most_common_interests_with(user_id):#한사람의 아이디를 넣어서, 그 사람과 같은 관심사를 가진 사람들 필터링하기.
#     l=[False for interest in interests_by_user_id["user_id"]#아이디별 관심도
#         for interested_user_id in user_ids_by_interest[interest]#관심도별 아이디
#         if interested_user_id != user_id]
#     print(f"interested_user_id:{l}")
# '''  
# # most_common_interests_with(users[3]["id"])


# #p10

# #1.3.3.연봉과 경력

# from collections import defaultdict

# salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
#                         (48000, 0.7), (76000, 6),
#                         (69000, 6.5), (76000, 7.5),
#                         (60000, 2.5), (83000, 10),
#                         (48000, 1.9), (63000, 4.2),(23000, 4.2)]

# '''
# #산점도 출력
# from matplotlib import pyplot as plt

# def make_chart_salaries_by_tenure():
#     tenures = [tenure for salary, tenure in salaries_and_tenures]
#     salaries = [salary for salary, tenure in salaries_and_tenures]
#     plt.scatter(tenures, salaries)
#     plt.xlabel("Years Experience")
#     plt.ylabel("Salary")
#     plt.show()
# '''
# make_chart_salaries_by_tenure()

# salary_by_tenure = defaultdict(list)

# for salary, tenure in salaries_and_tenures:
#     salary_by_tenure[tenure].append(salary)#경력별 연봉

# average_salary_by_tenure = { #딕셔너리 형태로, key-는 경력(연) , value는 평균 연봉
#     tenure : sum(salaries) / len(salaries)# 각 경력별의 총 합 / 명 수
#     for tenure, salaries in salary_by_tenure.items()
# }
# print(f"average_salary_by_tenure:{average_salary_by_tenure}")#동일 경력의 평균 연봉이 출력된다.
# #average_salary_by_tenure:{8.7: 83000.0, 8.1: 88000.0, 0.7: 48000.0, 6: 76000.0, 6.5: 69000.0, 7.5: 76000.0, 2.5: 60000.0, 10: 83000.0, 1.9: 48000.0, 4.2: 43000.0}

# def tenure_bucket(tenure):
#     if tenure < 2: return "less than two"
#     elif tenure < 5: return "between two and five"
#     else: return "more than five"
    
# salary_by_tenure_bucket = defaultdict(list)#경력구간 별 연봉

# for salary, tenure in salaries_and_tenures:
#     bucket = tenure_bucket(tenure)
#     salary_by_tenure_bucket[bucket].append(salary)

# # print(f"salary_by_tenure_bucket:{dict(salary_by_tenure_bucket)}")#수식어 삭제하기 위해선 dict() 자료형 변환 필요
# print(f"salary_by_tenure_bucket:{salary_by_tenure_bucket}")

# #p12

# #1.3.4 유료계정

# def predict_paid_or_unpaid(years_experience):
#     if years_experience < 3.0: 
#           return "paid"
#     elif years_experience < 8.5: 
#         return "unpaid"
#     else: 
#         return "paid"

# #p13

# #1.3.5 관심주제

# interests = [
#     (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
#     (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
#     (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
#     (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
#     (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
#     (3, "statistics"), (3, "regression"), (3, "probability"),
#     (4, "machine learning"), (4, "regression"), (4, "decision trees"),
#     (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
#     (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
#     (6, "probability"), (6, "mathematics"), (6, "theory"),
#     (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
#     (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
#     (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
#     (9, "Java"), (9, "MapReduce"), (9, "Big Data")
# ]

# from collections import Counter

# words_and_counts = Counter(word #관심도 조사에서 관심도를 가져온후
#                            for user, interest in interests
#                            for word in interest.lower().split())#소문자로 변환하고, 공백기준으로 나눠서 단어별 정리한다.

# print(words_and_counts)
# #Counter({'big': 3, 'data': 3, 'java': 3, 'python': 3, 'learning': 3, 'hadoop': 2, 'hbase': 2, 'cassandra': 2, 'scikit-learn': 2, 
# # 'r': 2, 'statistics': 2, 'regression': 2, 'probability': 2, 'machine': 2, 'neural': 2, 'networks': 2, 'spark': 1, 'storm': 1, 'nosql': 1, 
# # 'mongodb': 1, 'postgres': 1, 'scipy': 1, 'numpy': 1, 'statsmodels': 1, 'pandas': 1, 'decision': 1, 'trees': 1, 'libsvm': 1, 'c++': 1, 'haskell': 1, 
# # 'programming': 1, 'languages': 1, 'mathematics': 1, 'theory': 1, 'mahout': 1, 'deep': 1, 'artificial': 1, 'intelligence': 1, 'mapreduce': 1})


# for word, count in words_and_counts.most_common():#가장 많이 나온 순서대로 리스트 만들고, word와 count로 나눈다.
    
#     if count > 1:
#         print (word, count)
    
# print(type(words_and_counts.most_common()))
# print((words_and_counts.most_common()))