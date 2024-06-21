import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# from .attention import MultiHeadAttention

from .attention import PositionalEncoding, MultiHeadAttention

# Decoder layer implementation for a function

# 3 sublayers included

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    
    # # encoder의 self attention
    # enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask"
    )
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    # Fisrt sub_layer : Masked Multihead Self Attention
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask        
        })
    
    # LayerNormalization
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + inputs)
    
    # ------- 인코더 디코더 어텐션을 사용하지 않음 --------#
    # # Second sub_layer : Encoder-decoder Attention
    # attention2 = MultiHeadAttention(
    #     d_model, num_heads, name="attention_2")(inputs={
    #     'query': attention1,
    #     'key': enc_outputs,
    #     'value': enc_outputs,
    #     'mask': padding_mask
    #     })
    
    # Dropout, Layer Normalization(Residual connection)
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + inputs)
    
    # Third layer : 2 Fully-connected Layer
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        
    # Dropout, Layer Normalization(Residual connection)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2) 
    
    return tf.keras.Model(
        inputs=[inputs, 
                # enc_outputs, 
                look_ahead_mask, padding_mask], 
        outputs=outputs, 
        name=name
    )    
    
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')

    # 패딩 마스크
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 임베딩 레이어
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    # 포지셔널 인코딩
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    # Dropout이라는 훈련을 돕는 테크닉을 수행
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
            )(inputs=[outputs, 
                    #   enc_outputs, 
                      look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, 
                # enc_outputs, 
                look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
