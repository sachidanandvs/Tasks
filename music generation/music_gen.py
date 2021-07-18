import os
import json 
import numpy as np
import pandas as pd
import tensorflow as tf
#from tf.keras.models import Sequential
#from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
#from keras.callbacks import ModelCheckpoint
#from keras.utils import *
from music21 import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


data_directory = "/home/sachi/Documents/data"
data_file = "Data_Tunes.txt"
charIndex_json = "char_to_index.json"
BATCH_SIZE = 16
SEQ_LENGTH = 64

def preprocess(data):
  list1=list(data)
  list2=['\n','\n','\n']
  ignore=['X','T','M','S','K','P']
  i=0
  #to remove Part1:
  while(i<len(list1)):
    if(((list1[i] in ignore) and (list1[i+1]==":"))or list1[i]=='%' ):
      del list2[-1]
      while(list1[i]!='\n'):
        i=i+1
    list2.append(list1[i])
    i=i+1
  i=0
  #to append 'Z'(start token)
  preprocess_data=[]
  while(i<len(list2)):
    if(list2[i]=='\n'and list2[i+1]=='\n' and list2[i+2]=='\n'):
      preprocess_data.append('Z')
      i=i+3
    else:
      preprocess_data.append(list2[i])
      i=i+1
  return preprocess_data



def read_data(preprocess_data):
  char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(preprocess_data))))}

    
  with open(os.path.join(data_directory, charIndex_json), mode = "w") as f:
        json.dump(char_to_index, f)
        
  index_to_char = {i: ch for (ch, i) in char_to_index.items()}
  num_unique_chars = len(char_to_index)
  all_characters_as_indices = np.asarray([char_to_index[c] for c in preprocess_data], dtype = np.int32)
  return all_characters_as_indices,num_unique_chars

def input_output(all_chars_as_indices,num_unique_chars):
    total_length = all_chars_as_indices.shape[0]
    num_examples=int(total_length/SEQ_LENGTH)
    X=np.zeros((num_examples,SEQ_LENGTH))
    Y=np.zeros((num_examples,SEQ_LENGTH,num_unique_chars))
    for i in range(num_examples):
      for j in range(SEQ_LENGTH):
        X[i,j]=all_chars_as_indices[i*SEQ_LENGTH+j]
        Y[i,j,all_chars_as_indices[i*SEQ_LENGTH+j+1]]=1
    return X,Y

def build_model( seq_length, num_unique_chars):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Embedding(input_dim = num_unique_chars, output_dim = 512, input_shape = (seq_length,))) 
    
    model.add(tf.keras.layers.LSTM(256, return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.LSTM(256, return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(256, return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_unique_chars)))

    model.add(tf.keras.layers.Activation("softmax"))
    
    return model

def make_model(num_unique_chars):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Embedding(input_dim = num_unique_chars, output_dim = 512, batch_input_shape = (1, 1))) 
  
# stateful: If True, the last state for each sample at index i in a batch will be used 
# as initial state for the sample of index i in the following batch.
    model.add(tf.keras.layers.LSTM(256, return_sequences = True, stateful = True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.LSTM(256, return_sequences = True, stateful = True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.LSTM(256,return_sequences=True, stateful = True)) 
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add((tf.keras.layers.Dense(num_unique_chars)))
    model.add(tf.keras.layers.Activation("softmax"))
    
    return model

def generate_sequence(gen_seq_length):
    with open(os.path.join(data_directory, charIndex_json)) as f:
        char_to_index = json.load(f)
    index_to_char = {i:ch for ch, i in char_to_index.items()}
    num_unique_chars = len(index_to_char)
    
    model = make_model(num_unique_chars)
    model.load_weights("/home/sachi/Documents/weights.80.hdf5")
     
    sequence_index = [char_to_index['Z']]

    for _ in range(gen_seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(num_unique_chars), size = 1, p = predicted_probs)
        
        
        sequence_index.append(sample[0])
    
        
    
    seq = ''.join(index_to_char[c] for c in sequence_index)
    seq='M:6/8\n'+str(seq)
    return seq

def convert_to_midi(abc):
    c = converter.subConverters.ConverterABC()
    c.registerOutputExtensions = ("midi", )
    c.parseData(abc)
    s = c.stream
    s.write('midi', fp='demos1.mid')

file = open(os.path.join(data_directory, data_file), mode = 'r')
data = file.read()
file.close()
preprocess_data=preprocess(data)
all_characters_as_indices,num_unique_chars=read_data(preprocess_data)
X,Y=input_output(all_characters_as_indices,num_unique_chars)
print("length of preprocess_data-{}".format(len(preprocess_data)))
print("vocab_size={}".format(num_unique_chars))
print("all_characters={}".format(all_characters_as_indices))
print("length of all_characters-{}".format(len(all_characters_as_indices)))
print("shape of X={}".format(X.shape))
print("shape of Y={}".format(Y.shape))


model=build_model(SEQ_LENGTH,num_unique_chars)
model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath='weights.{epoch:02d}.hdf5',monitor='loss',save_best_only=True,save_weights_only=True,period=10)
model.fit(X,Y,batch_size=16,epochs=80,callbacks=[checkpoint])

music = generate_sequence(192)
print("\nMUSIC SEQUENCE GENERATED: \n{}".format(music))
convert_to_midi(music)