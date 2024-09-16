import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def truncated_normal_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(mean=0., stddev=stddev)



#without RPE
class NormalAttention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(NormalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dimension must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=truncated_normal_initializer())
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, kernel_initializer=truncated_normal_initializer())
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, -1, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (tf.matmul(q, k, transpose_b=True)) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, -1, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

# general mapping
class GeneralAttention(layers.Layer):
    def __init__(self, lengthofside, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale_factor = 10, multihead = True):
        super(GeneralAttention, self).__init__()
        self.height = lengthofside
        self.width = lengthofside

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dimension must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=truncated_normal_initializer())
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, kernel_initializer=truncated_normal_initializer())
        self.proj_drop = layers.Dropout(proj_drop)

        max_distance = tf.sqrt(tf.square(lengthofside) + tf.square(lengthofside - 1))
        self.scale_factor = scale_factor
        scaled_max_distance = tf.cast(max_distance * scale_factor, tf.int32)
        self.multihead = multihead
        if multihead:
            self.distance_mappings = [
                tf.Variable(
                    initial_value=tf.random.uniform(shape=[scaled_max_distance + 1], minval=-1, maxval=1),
                    trainable=True
                )
            for _ in range(num_heads)
        ]
        else:
            self.distance_mapping = tf.Variable(
                initial_value=tf.random.uniform(shape=[scaled_max_distance + 1], minval=-1, maxval=1), trainable=True)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, -1, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        indices = tf.range(self.height * self.width)  # Start from 1 to reserve 0 for cls_token
        grid_y = indices // self.width + 1
        grid_x = indices % self.width
        grid_positions = tf.stack([grid_y, grid_x], axis=1)
        cls_pos = tf.cast(tf.constant([[0, 0]]) , tf.float32) # Position cls_token at the top-left
        grid_positions = tf.concat([cls_pos, grid_positions], axis=0)

        # Compute all pairwise Euclidean distances
        diffs = tf.expand_dims(grid_positions, axis=1) - tf.expand_dims(grid_positions, axis=0)
        distances = tf.norm(tf.cast(diffs, dtype=tf.float32), axis=-1)


        attn = (tf.matmul(q, k, transpose_b=True)) * self.scale

        if self.multihead:
            Mij_list = []
            for head_idx in range(self.num_heads):
                distance_indices = tf.cast(distances * self.scale_factor, tf.int32)
                Mij = tf.gather(self.distance_mappings[head_idx], distance_indices)
                Mij_list.append(Mij)
            Mij = tf.stack(Mij_list, axis=0)
            attn = attn * Mij[None, :, :, :]
        else:
            # Map distances through the distance mapping tensor
            distance_indices = tf.cast(distances * self.scale_factor, tf.int32)
            Mij = tf.gather(self.distance_mapping, distance_indices)
            attn = attn * Mij[None, None, :, :]

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, -1, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x




# exponential version
class ExponentialAttention(layers.Layer):
    def __init__(self, lengthofside, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., multihead = True):
        super(ExponentialAttention, self).__init__()
        self.height = lengthofside
        self.width = lengthofside

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dimension must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=truncated_normal_initializer())
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, kernel_initializer=truncated_normal_initializer())
        self.proj_drop = layers.Dropout(proj_drop)

        self.multihead = multihead

        if multihead:
            self.exp_param = [
                tf.Variable(initial_value=tf.random.normal(shape=[], mean=0.0, stddev=0.1), trainable=True)
                for _ in range(num_heads)
            ]
        else:
            self.exp_param = tf.Variable(initial_value=tf.random.normal(shape=[], mean=0.0, stddev=0.1), trainable=True)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, -1, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        indices = tf.range(self.height * self.width)  # Start from 1 to reserve 0 for cls_token
        grid_y = indices // self.width + 1
        grid_x = indices % self.width
        grid_positions = tf.stack([grid_y, grid_x], axis=1)
        cls_pos = tf.cast(tf.constant([[0, 0]]) , tf.float32) # Position cls_token at the top-left
        grid_positions = tf.concat([cls_pos, grid_positions], axis=0)

        # Compute all pairwise Euclidean distances
        diffs = tf.expand_dims(grid_positions, axis=1) - tf.expand_dims(grid_positions, axis=0)
        distances = tf.norm(tf.cast(diffs, dtype=tf.float32), axis=-1)

        attn = (tf.matmul(q, k, transpose_b=True)) * self.scale

        if self.multihead:
            Mij_list = []
            for head_idx in range(self.num_heads):
                Mij = tf.exp(-self.exp_param[head_idx] * distances)
                Mij_list.append(Mij)
            Mij = tf.stack(Mij_list, axis=0)
            attn = attn * Mij[None, :, :, :]
        else:
            Mij = tf.exp(-self.exp_param * distances)
            attn = attn * Mij[None, None, :, :]


        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, -1, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


# the ratio of polynomial
class PolynomialAttention(layers.Layer):
    def __init__(self, lengthofside, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., degree = 3, multihead = True):
        super(PolynomialAttention, self).__init__()
        self.height = lengthofside
        self.width = lengthofside

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dimension must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=truncated_normal_initializer())
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, kernel_initializer=truncated_normal_initializer())
        self.proj_drop = layers.Dropout(proj_drop)

        self.degree = degree
        self.multihead = multihead

        if multihead:
            self.poly_coeffs = [[tf.Variable(initial_value=tf.random.normal(shape=[], mean=0.0, stddev=0.1), trainable=True) for _ in range(degree + 1)]
            for _ in range(num_heads)]
        else:
            self.poly_coeffs = [tf.Variable(initial_value=tf.random.normal(shape=[], mean=0.0, stddev=0.1), trainable=True) for _ in range(degree + 1)]


    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, -1, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        indices = tf.range(self.height * self.width)  # Start from 1 to reserve 0 for cls_token
        grid_y = indices // self.width + 1
        grid_x = indices % self.width
        grid_positions = tf.stack([grid_y, grid_x], axis=1)
        cls_pos = tf.cast(tf.constant([[0, 0]]) , tf.float32) # Position cls_token at the top-left
        grid_positions = tf.concat([cls_pos, grid_positions], axis=0)

        # Compute all pairwise Euclidean distances
        diffs = tf.expand_dims(grid_positions, axis=1) - tf.expand_dims(grid_positions, axis=0)
        distances = tf.norm(tf.cast(diffs, dtype=tf.float32), axis=-1)

        attn = (tf.matmul(q, k, transpose_b=True)) * self.scale

        if self.multihead:
            Mij_list = []
            for head_idx in range(self.num_heads):
                Mij =  1.0 / tf.math.polyval(self.poly_coeffs[head_idx], distances + 1.0)
                Mij_list.append(Mij)
            Mij = tf.stack(Mij_list, axis=0)
            attn = attn * Mij[None, :, :, :]
        else:
            Mij = 1.0 / tf.math.polyval(self.poly_coeffs, distances + 1.0)
            attn = attn * Mij[None, None, :, :]

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, -1, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


