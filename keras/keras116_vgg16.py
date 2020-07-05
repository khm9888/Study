from keras.applications import VGG16,VGG19,DenseNet121,DenseNet169,DenseNet201,MobileNet,Xception
from keras.applications import MobileNetV2,NASNetLarge, NASNetMobile,InceptionResNetV2,InceptionV3,ResNet101,ResNet152V2
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization,Activation
from keras.models import Sequential
from keras.applications import ResNet101V2
from keras.applications import ResNet152
from keras.applications import ResNet50
from keras.applications import ResNet50V2
from keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam


# print(cifar10.load_data().shape)
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#1) 데이터 구성

print(f"x_test.shape:{x_test.shape}")#x_test.shape:(50000, 32, 32, 3)
print(f"y_test.shape:{y_test.shape}")#y_test.shape:(50000, 1)

#dimension 맞춰주기 (4->2)

x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

#scaler 처리

scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#dimension 맞춰주기 (2->4)

x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)

#y값에 대해서 to_categorical() 해주기

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

print(f"y_test.shape:{y_test.shape}")

# model = VGG19()
# model = Xception()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# model = NASNetMobile()

# def build_model_vgg16(optimizer=adam,node=32,loss="sparse_categorical_crossentropy"):
#     model=Sequential()
#     model.add(VGG16(include_top=False,weights="imagenet",input_tensor=(32,32,3)))
#     model.add(Flatten())
#     model.add(Dense())
#     model.add(BatchNormalization())
#     model.add(Activation("relu"))
#     model.add(Dense(10,activation="softmax"))

#     model.compile(loss=loss,optimizer=optimizer,metrics=['acc'])
#     model.fit(x_train,y_train,batch_size=100,epochs=20)
#     return model

# model =build_model_vgg16()

model=Sequential()
vgg=VGG16(include_top=False,weights="imagenet",input_shape=(32,32,3))
model.add(vgg)
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(10,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",optimizer=adam(), metrics=['acc'])
model.fit(x_train,y_train,batch_size=100,epochs=20,validation_split=0.3)

loss,acc = model.evaluate(x_test,y_test)
print("loss")
print(loss)
print("acc")
print(acc)
