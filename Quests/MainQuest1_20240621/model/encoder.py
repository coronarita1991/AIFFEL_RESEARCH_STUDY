import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# encoder layer implementation for a function

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    # First layer : Multi Head Self Attention 
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    
    # Dropout, Layer Normalization(Residual connection)
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention) 
    # epsilon -  Small float added to variance to avoid dividing by zero. Defaults to 1e-3. 
    
    # Second layer : 2 Fully-connected Layer
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        
    # Dropout, Layer Normalization(Residual connection)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs) 
    
    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name
    )
    
def encoder(vocab_size,
           num_layers,
           units,
           d_model,
           num_heads,
           dropout,
           name="encoder"):
    
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    # Embedding layer
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # Scale Normalizing for positional encoding
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32)) 
    
    # Positional encoding
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    
    # Encoder layer for num_layers
    for i in range(num_layers):
        outputs = encoder_layer(
                units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])
        
    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name
        
    )
        