import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Conv2D,\
MaxPooling2D, GlobalAveragePooling2D 
from features.RID_SPIC22 import *

def RID_SPIC_model():
    inputs = Input(shape=(900, 600, 3)) 
    outputs = Lambda(RID_SPIC22, output_shape=(324,))(inputs)
    
    model = Model(inputs, outputs)
    return model
