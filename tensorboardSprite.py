import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
from flask import Flask, send_from_directory
import subprocess


app = Flask(__name__)


@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


def launch_tensorboard():
    subprocess.Popen(["tensorboard", "--logdir=."])


if __name__ == '__main__':
    vectors_path = 'similarity_matrixE.tsv'
    vectors = []
    with open(vectors_path, 'r') as f:
        for line in f:
            vector = line.strip().split('\t')
            vector = [float(value) for value in vector]
            vectors.append(vector)
    vectors = np.array(vectors)

    metadata_path = 'metadata.tsv'
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(line.strip())

    # Create a sprite image from individual images
    sprite_rows = 4  # Number of rows in the sprite grid
    sprite_cols = 4  # Number of columns in the sprite grid
    sprite_width = 100  # Width of each individual image in the sprite
    sprite_height = 100  # Height of each individual image in the sprite

    # Create a checkpoint from embedding vectors.
    weights = tf.Variable(vectors)
    checkpoint = tf.train.Checkpoint(embedding=weights)

    # Set up config.
    config = projector.ProjectorConfig()

    # Add the embedding tensor to the config.
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding"

    # Configure sprite image
    embedding.sprite.image_path = 'sprite.png'
    embedding.sprite.single_image_dim.extend([sprite_width, sprite_height])

    # Specify the path to your metadata file.
    embedding.metadata_path = 'metadata.tsv'

    # Save the metadata file.
    with open(metadata_path, 'w') as f:
        for line in metadata:
            print(line)
            f.write(line + '\n')

    # Save the checkpoint in the default path
    checkpoint.save("embedding.ckpt")

    # Save the projector config file.
    projector.visualize_embeddings('.', config)

    # Launch TensorBoard using subprocess
    launch_tensorboard()
    print(embedding.metadata_path)
    app.run()
