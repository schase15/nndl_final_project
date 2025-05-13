#Run the google colab notebook that will host CLIP server.
# Make sure you shut it down after with f.close()

from clip_client import Client
from docarray import Document
import os
import base64

# Grab the tcp url from the running colab.
# !ngrok tcp 51000 --log "stdout"
# Grab the part after url=tcp://
tcp_url = '0.tcp.ngrok.io:13104'
c = Client('grpc://' + tcp_url)

# Load and encode a single image
image_path = os.path.join('data/Released_Data_NNDL_2025/train_images', '0.jpg')

# Read image and convert to base64
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    image_uri = f'data:image/jpeg;base64,{base64_image}'

# Get embedding
embedding = c.encode([image_uri])
print(f"Embedding shape: {embedding.shape}")
print(f"First 10 values: {embedding[:10]}")
