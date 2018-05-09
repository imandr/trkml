import keras
from keras import backend as K
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Bidirectional
from keras.optimizers import Adadelta

def create_model_bidirectional():
    
    inp = Input(shape=(None, 3))
    
    lstm_bidir = Bidirectional(LSTM(30, return_sequences=True, implementation=2))(inp)

    dense1 = Dense(16)(lstm_bidir)
    
    model = Model(inputs=[inp], outputs=[dense1])
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model
    
def create_model_back_forth():
    
    inp = Input(shape=(None, 3))
    
    inp_rev = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (inp)
    
    lstm1 = LSTM(30, return_sequences=True)(inp)

    mix = keras.layers.concatenate([lstm1, inp])
    lstm2 = LSTM(30, return_sequences=True, go_backwards=True)(mix)

    lstm2_rev = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm2)
    
    dense1 = Dense(16)(lstm2_rev)
    
    model = Model(inputs=[inp], outputs=[dense1])
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model
    
def create_model_0():
    
    inp = Input(shape=(None, 3))
    
    dense1 = Dense(16, activation="sigmoid")(inp)
    
    model = Model(inputs=[inp], outputs=[dense1])
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

if __name__ == '__main__':

    model = create_model()

    x = np.random.random((1, 20, 3))

    y = model.predict(x)
   
    print y.shape 
    
