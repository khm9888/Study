#Load

from keras.models import load_model,Model
from keras.layers import Dense

model=load_model("deeplearning.h5")

# output=(Dense(1,name="dense_x"))(dense_5)

model.summary()
