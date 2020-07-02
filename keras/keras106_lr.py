weight = 0.5
input_value = 1.5
goal_prediction=0.8

lr = 0.1


prediction = input_value*weight #0.5*0.5 = 0.25
error = (prediction-goal_prediction)**2 #(0.25 - 0.8)**2 -> 오차값

# # 결과 텍스트 파일로 저장하기 - acc
name=__file__.split("\\")[-1]
path=__file__[:-len(name)]+"results/"+f"{name}.txt"

#새로 쓸지, 아니면 추가할지 _ 없으면 새로 생성한다.
for iteration in range(10):
    mode="a"
    with open(path,mode,encoding="utf-8") as file:
        
        file.write("\n")
        file.write("*"*20+"\n") 

        file.write(f"{iteration}")
        file.write("\n")
        file.write("error")
        file.write("\n")
        file.write(f"{error}")
        file.write("\n")
        file.write("prediction")
        file.write("\n")
        file.write(f"{prediction}")
        file.write("\n")

        up_prediction = input_value*(weight+lr)#weigth>lr>0 이면, up_prediction은 점차 커진다.
        up_error = (goal_prediction-up_prediction)**2 

        down_prediction = input_value*(weight-lr)#weigth>lr>0 이면, up_prediction은 점차 작아진다.
        down_error = (goal_prediction-down_prediction)**2

        
        file.write("up_prediction")
        file.write("\n")
        file.write(f"{up_prediction}")
        file.write("\n")
        file.write("down_prediction")
        file.write("\n")
        file.write(f"{down_prediction}")
        file.write("\n")

        if (down_error<up_error):# down_prediction > up_prediction 이라는 뜻
            
            file.write("case1")
            file.write("\n")
            weight = weight-lr
        elif down_error>up_error:# down_prediction < up_prediction 이라는 뜻
            
            file.write("case2")        
            file.write("\n")
            weight = weight+lr

        file.write("*"*20+"\n")
        file.write("\n")
        