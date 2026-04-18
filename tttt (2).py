import tensorflow as tf
import numpy as np
import pandas as pd
import random
d=pd.read_csv('/content/train.csv',on_bad_lines='skip')
text=" ".join(d['text'].dropna().astype(str)).lower()
print(f'Total characters in text: {len(text)}')
vocab=sorted(set(text))
print(f'Vocabulary size: {len(vocab)}')
char2idx={c:i for i,c in enumerate(vocab)}
idx2char=np.array(vocab)
text_as_int=np.array([char2idx[c] for c in text])
seq_length=100
char_dataset=tf.data.Dataset.from_tensor_slices(text_as_int)
sequences=char_dataset.batch(seq_length+1,drop_remainder=True)
def split_input_target(chunk):
    return chunk[:-1],chunk[1:]
dataset=sequences.map(split_input_target)
BATCH_SIZE=64
BUFFER_SIZE=10000
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
vocab_size=len(vocab)
embedding_dim=64
rnn_units=128
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_shape=(None,)),
    tf.keras.layers.LSTM(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
def loss(labels,logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
model.compile(optimizer='adam',loss=loss)
model.summary()
EPOCHS=20
history=model.fit(dataset,epochs=EPOCHS)
def generate_text(model,start_string,num_generate=100,temperature=1.0):
    input_eval=[char2idx.get(s,0) for s in start_string.lower()]
    input_eval=tf.expand_dims(input_eval,0)
    text_generated=[]
    model.layers[1].reset_states()  
    for i in range(num_generate):
        predictions=model(input_eval)
        predictions=tf.squeeze(predictions,0) / temperature
        predicted_id=tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

        input_eval=tf.expand_dims([predicted_id],0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)
print(generate_text(model,start_string="The ",num_generate=200,temperature=0.8))




