'''
This script contains the methods to build a Transformers based decoder architecture 
'''
import tensorflow as tf
import keras
from keras import layers
import keras_nlp
import numpy as np

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

def get_model(shape, sequence_length, embed_dim=1, num_heads=6, num_layers=1, dropout_rate=0.5):
    
    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = keras_nlp.layers.TransformerEncoder(embed_dim, num_heads, name="transformer_layer")(x)
    for i in range(num_layers - 1):
        x = keras_nlp.layers.TransformerEncoder(embed_dim, num_heads, name=f"transformer_layer{i}")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))(x)
    model = keras.Model(inputs, outputs)
    return model