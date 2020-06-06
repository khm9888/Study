# 프로그래밍
# 1번 문제
# 아래 예시의 JSON 포맷을 매핑하는 Male, Female 모델 클래스를 정의 및 구현하세요. 그리고
# 매핑할 때 각 키마다 우측 주석의 유효성을 체크하여 잘못된 값이면 예외 처리하고 에러
# 메시지를 출력하세요. 모든 유효성 검사를 통과하면 입력 값을 전부 출력하세요. JSON은
# 파일이나 문자열로 입력 받습니다. (배점 30)
# [입력 값 - Male]
# {
# “name” : “민호” // 이름 : 한글 2, 3 글자만 가능
# “birthday” : “1987-06-22” // 생년월일 : yyyy-MM-dd 포맷 정확히
# 일치해야함
# “height” : 187 // 키 : 160~200 범위의 숫자만 가능
# “job” : “배우” // 직업 : 한글, 영문 대소문자 가능, 10 자
# 제한
# “location” : “126.9683104,37.344701” // 위치 : 경도, 위도 좌표 포맷
# “hobbies” : [“독서”, “영화”, “축구”] // 취미 : 해당 키가 없거나(null), 0 개일 수
# 있음
# “is_fulfilled” : false // 군필 여부 : true or false
# }
# [입력 값 - Female]
# {
# “name” : “수지” // 이름 : 한글 2, 3 글자만 가능
# “birthday” : “1994-10-10” // 생년월일 : yyyy-MM-dd 포맷 정확히
# 일치해야함
# “height” : 168 // 키 : 150~190 범위의 숫자만 가능
# “job” : “배우” // 직업 : 한글, 영문 대소문자 가능, 10 자
# 제한
# “location” : “127.0292881,37.5108295” // 위치 : 경도, 위도 좌표 포맷
# “hobbies” : [“봉사활동”, “영화”, “독서”] // 취미 : 해당 키가
# 없거나(null),
# 0 개일 수 있음
# }

# import json

# class male:
# 	information={}
# 	def __init__(self,name,birthday,height,job,location,hobbies,is_fulfilled):
# 		if len(name)==2 or len(name)==3:
# 			self.name.information["name"]=name
# 		self.birthday.information["birthday"]=birthday
# 		if 160<=height<=200:
# 			self.height.information["height"]=height
# 		if len(job)<=10:
# 			self.job.information["job"]=job
# 		#경도 위치 포맷
# 		self.location.information["location"]=location
# 		if hobbies==null:
# 			hobbies==""
# 		else:
# 			self.hobbies.information["hobbies"]=hobbies
# 		if is_fulfilled:
# 			self.is_fulfilled.information["is_fulfilled"]=is_fulfilled
# 	def_setdata():
# 		for i in range(7):
# 			input()
# 	def info_print():
# 		print(information)

# class female:
# 	information={}
# 	def __init__(self,name,birthday,height,job,location,hobbies,is_fulfilled):
# 		self.name.information["name"]=name
# 		self.birthday.information["birthday"]=birthday
# 		self.height.information["height"]=height
# 		self.job.information["job"]=job
# 		self.location.information["location"]=location
# 		self.hobbies.information["hobbies"]=hobbies
# 	def info_print():
# 		print(information)

		
import re
from scipy.spatial import distance
import json


class Male_Info:
	
	def __init__(self,name,birthday,height,job,location,hobbies,is_fulfilled):
		self.check=True
		
		if len(name)==2 or len(name)==3:
			self.name=name
		else:
			self.check=False
			print("이름의 글자수를 2~3글자로 입력해주세요.")

		
		if bool(re.match('\d{4}-\d{2}-\d{2}',birthday)):
			self.birthday=birthday
		else:
			self.check=False
			print("생년월일을 형식에 맞게 입력해주세요")

		
		if 160<=height<=200:
			self.height=height
		else:
			self.check=False
			print("키는 160~200 사이가 되어야합니다.")

		
		if len(job)<=10:
			self.job=job
		else:
			self.check=False
			print("직업은 10자 이내로 작성해주세요.")

		
		if bool(re.match('^(\-?\d+(\.\d+)?),\s*(\-?\d+(\.\d+)?)$',location)):
			self.location=location
		else:
			self.check=False
			print("좌표 입력 형식이 잘못되었습니다.")

		if hobbies==[]:
			self.hobbies=[]
		else:
			self.hobbies=hobbies

		
		if is_fulfilled==True or is_fulfilled==False:
			self.is_fulfilled=is_fulfilled

		else:
			check=False
			print("군필유무를 True/False 로 입력해주세요")


		#name,birthday,height,job,location,hobbies,is_fulfilled
	def save(self):
		information=dict()
		information["name"]=self.name
		information["birthday"]=self.birthday
		information["height"]=self.height
		information["job"]=self.job
		information["location"]=self.location
		information["hobbies"]=self.hobbies
		information["is_fulfilled"]=self.is_fulfilled

		if self.check:
			# print(information)
			t1=json.dumps(str(information))
			print(json.loads(t1))
		return t1

class Female_Info:
	
	def __init__(self,name,birthday,height,job,location,hobbies):
		self.check=True
		
		if len(name)==2 or len(name)==3:
			self.name=name
		else:
			self.check=False
			print("이름의 글자수를 2~3글자로 입력해주세요.")

		
		if bool(re.match('\d{4}-\d{2}-\d{2}',birthday)):
			self.birthday=birthday
		else:
			self.check=False
			print("생년월일을 형식에 맞게 입력해주세요")

		
		if 160<=height<=200:
			self.height=height
		else:
			self.check=False
			print("키는 160~200 사이가 되어야합니다.")

		
		if len(job)<=10:
			self.job=job
		else:
			self.check=False
			print("직업은 10자 이내로 작성해주세요.")

		
		if bool(re.match('^(\-?\d+(\.\d+)?),\s*(\-?\d+(\.\d+)?)$',location)):
			self.location=location
		else:
			self.check=False
			print("좌표 입력 형식이 잘못되었습니다.")

		if hobbies==[]:
			self.hobbies=[]
		else:
			self.hobbies=hobbies


		#name,birthday,height,job,location,hobbies,is_fulfilled
	def save(self):
		information=dict()
		information["name"]=self.name
		information["birthday"]=self.birthday
		information["height"]=self.height
		information["job"]=self.job
		information["location"]=self.location
		information["hobbies"]=self.hobbies

		if self.check:
			# print(information)
			t1=json.dumps(str(information))
			print(json.loads(t1))
		return t1

class CoupleManager:
	
	def __init__(self, p1,p2):
		self.p1=p1
		self.p2=p2
		

	def age_gap(self):
		this_year=2020

		p1_age=this_year-int(self.p1.birthday[0:4])+1
		p2_age=this_year-int(self.p2.birthday[0:4])+1
		
		if p1_age<p2_age:
			gap="연상"
			print(self.p2.name[1:]+"님은 "+str(abs(p1_age-p2_age))+"살 " +gap+"입니다" )
		elif p1_age>p2_age:
			gap="연하"
			print(self.p2.name[1:]+"님은 "+str(abs(p1_age-p2_age))+"살 " +gap+"입니다" )
		else:
			gap="동갑"
			print(self.p2.name[1:]+"님은 당신과 "+gap+"입니다" )

	def distant(self):
		v1=list(map(float,self.p1.location.split(",")))
		v2=list(map(float,self.p2.location.split(",")))
		v3=int(distance.euclidean(v1,v2)*100,)
		if v3<=5:
			print(self.p2.name[1:]+"님은 5km 이내에 위치하고 있습니다")
		else:
			print(self.p2.name[1:]+"님은 "+str(v3)+"km 거리에 위치하고 있습니다")
		#print(,"km 떨어져있습니다.")

	def count_hobby(self):
		count=0
		for h in self.p1.hobbies:
			if h in	self.p2.hobbies:
				count+=1

		print(self.p2.name[1:]+"님은 "+str(count)+"개의 취미가 겹칩니다.")

	def army(self):
		if hasattr(self.p2,"is_fulfilled"):
			if self.p2.is_fulfilled:
				print(self.p2.name[1:]+"님은 군필자입니다.")
			else:
				print(self.p2.name[1:]+"님은 미필자입니다.")


man=Male_Info("김해민","1991-04-28",176,"프리랜서","127.0292881,37.5108295",["헬스","보드게임","코노","독서"],False)
woman=Female_Info("김여자","1990-04-28",176,"프리랜서","127.0292881,37.5208295",["헬스","보드게임","요가"])

man.save()
woman.save()

c = CoupleManager(woman,man)

c.age_gap()
c.distant()
c.count_hobby()
c.army()

