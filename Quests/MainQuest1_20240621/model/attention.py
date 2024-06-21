import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt


# Scaled Dot Product Attention
def scaled_dot_product_attention(query, key, value, mask): # (seq_len, depth)
    # Attention weight
    matmul_qk = tf.matmul(query, key, transpose_b=True) # (seq_len, seq_len)
    
    # Normalization
    depth = tf.cast(tf.shape(key)[-1], tf.float32) # depth = d_model / num_heads
    logits = matmul_qk / tf.math.sqrt(depth) # (seq_len, seq_len)
    
    # Add mask to Padding
    if mask is not None: 
        logits += (mask * -1e9)
        
    # softmax
    attention_weights = tf.nn.softmax(logits, axis=-1) # query에 대한 key의 softmax값 
    
    # Scaled Dot Product
    output = tf.matmul(attention_weights, value) # (seq_len, seq_len) x (seq_len, depth)
    return output # (seq_len, depth)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model) # (1, position, d_model), dtype=float32 
    
    # pos/10000^(2i/d_model) 를 반환
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, 
                            (2 * (i // 2)) / tf.cast(d_model, tf.float32)
                           )
        return position * angles # (position, d_model)
        
    def positional_encoding(self, position, d_model):
        # 각도 배열 생성
        angle_rads = self.get_angles(
            # tf.newaxis : 기존 텐서에 새 차원을 추가
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis], # (position, 1)
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], # (1, d_model)
            d_model=d_model) 
        # angle_rads : (position, d_model)
        
        # 배열의 짝수 인덱스 : sin함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2]) # 짝수번쨰(0, 2, 4, ....) : (position, d_model/2)
        # 배열의 홀수 인덱스 : cos함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2]) # 홀수번쨰(1, 3, 5, ....) : (position, d_model/2)
        
        # sine, cosine이 교차되도록 재배열
        pos_encoding = tf.stack([sines, cosines], axis=0) # 각 연산이 된 결과들을 쌓는다. : (2, position, d_model/2)
        pos_encoding = tf.transpose(pos_encoding, [1, 2, 0]) # 순서를 바꾼다. : (position, d_model/2, 2)
        pos_encoding = tf.reshape(pos_encoding, [position, d_model]) # (position, d_model)
        
        pos_encoding = pos_encoding[tf.newaxis, ...] # (1, position, d_model)
        return tf.cast(pos_encoding, tf.float32) # type casting
    
    def call(self, inputs): # inputs을 (batch_size, seq_len, d_model) 이라고 하면(임베딩 후 결과)
        
        # [:, :tf.shape(inputs)[1], :] : slice (positional encoding) 
        # -> (batch_size, seq_len, d_model)
        
        # inputs에 의거, broadcasting이 일어남(self.pos_encoding이 inputs 배치에 합연산)
        
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        
        self.dense = tf.keras.layers.Dense(units=d_model)
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        # to use scaled_dot_product_attention
        return tf.transpose(inputs, perm=[0, 2, 1, 3]) 
        
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        # Q, K, V - Dense # (bs, seq_len, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
         
        # Parallel Computing - Multi heads : (batch_size, num_heads, seq_len, depth)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # scaled_attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len, num_heads, depth)
        
        # concatenate
        concat_attention = tf.reshape(scaled_attention,
                                     (batch_size, -1, self.d_model)) # (batch_size, seq_len, d_model)
        # Dense for final
        outputs = self.dense(concat_attention)
        
        return outputs