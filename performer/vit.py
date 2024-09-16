import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from .fast_attention import SelfAttention, SelfMaskedAttention

def truncated_normal_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(mean=0., stddev=stddev)

class DropPath(layers.Layer):
    """ Drop paths (stochastic depth) per sample. """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob is None or self.drop_prob == 0.:
            return x
        input_shape = tf.shape(x)
        random_tensor = tf.random.uniform(input_shape, dtype=x.dtype, minval=0, maxval=1)
        keep_prob = 1 - self.drop_prob
        binary_tensor = tf.floor(keep_prob + random_tensor)  # Keep if random tensor < keep_prob
        output = tf.divide(x, keep_prob) * binary_tensor
        return output

class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, activation=tf.keras.activations.gelu, kernel_initializer=truncated_normal_initializer())
        self.drop1 = layers.Dropout(drop)
        self.fc2 = layers.Dense(out_features, kernel_initializer=truncated_normal_initializer())
        self.drop2 = layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x

class PatchEmbedding(layers.Layer):
    def __init__(self, image_size, patch_size, num_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = layers.Dense(embed_dim)

    def call(self, images):

        # Create patches
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')


        # Flatten patches
        flat_patches = tf.reshape(patches, shape=[batch_size, self.num_patches, -1])

        # Embed patches
        embedded_patches = self.projection(flat_patches)
        return embedded_patches


class VisionTransformer(Model):
    def __init__(self, image_size, patch_size, num_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate, projection_matrix_type, nb_random_features,masked, mask_method, multihead,scale_factor):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim  # Make sure this is defined
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(image_size, patch_size, num_channels, embed_dim)
        self.cls_token = self.add_weight("cls_token", shape=(1, 1, embed_dim), initializer="zeros")
        self.pos_embed = self.add_weight("pos_embed", shape=(1, 1 + self.patch_embedding.num_patches, embed_dim), initializer="random_normal")
        self.dropout = layers.Dropout(drop_rate)

        self.transformer_blocks = [TransformerBlock(image_size / patch_size, embed_dim, num_heads, mlp_ratio, drop_rate, projection_matrix_type, nb_random_features,masked,mask_method,multihead,scale_factor) for _ in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs):

        x = self.patch_embedding(inputs)
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        # delete pos_embed
        x += self.pos_embed
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.classifier(x[:, 0])  # Using the representation of the cls token
        return x


'''
attention:
               (hidden_size,
               num_heads,
               attention_dropout)
'''
class TransformerBlock(layers.Layer):
    def __init__(self, length_of_side, dim, num_heads, mlp_ratio, drop_rate, projection_matrix_type, nb_random_features, masked, mask_method, multihead, scale_factor, drop_path_rate=0.1):
        super(TransformerBlock, self).__init__()
        if masked:
            self.attention = SelfMaskedAttention(hidden_size=dim, num_heads = num_heads, length_of_side = length_of_side, attention_dropout=drop_rate, projection_matrix_type=projection_matrix_type,
                   nb_random_features=nb_random_features, mask_method = mask_method, scale_factor = scale_factor, multihead = multihead)
        else:
            self.attention = SelfAttention(hidden_size=dim, num_heads=num_heads,
                                                 attention_dropout=drop_rate,
                                                 projection_matrix_type=projection_matrix_type,
                                                 nb_random_features=nb_random_features)

        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, out_features=dim, drop=drop_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        y = self.norm1(x)
        y = self.attention(y, training=training)
        x = x + self.drop_path(y, training=training)  # Residual connection

        y = self.norm2(x)
        y = self.mlp(y, training=training)
        x = x + self.drop_path(y, training=training)

        return x
