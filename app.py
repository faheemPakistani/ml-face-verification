from flask import Flask, render_template, request
import numpy as np
import cv2
import keras
import tensorflow as tf

# ===============

app = Flask(__name__)


def triplet_loss(y_true, y_pred, alpha=0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum((anchor - positive)**2, axis=-1)
    neg_dist = tf.reduce_sum((anchor - negative)**2, axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


triplet_model = keras.models.load_model(
    './face_verification.h5', custom_objects={'triplet_loss': triplet_loss})

# ======================


# def verify(image_path, identity, database, model):

#     encoding = img_to_encoding(image_path, model)

#     dist = np.linalg.norm(encoding-database[identity])

#     if dist < 0.7:
#         print("It's " + str(identity) + ", welcome!")
#         door_open = True
#     else:
#         print("It's not " + str(identity) + ", please go away")
#         door_open = False

#     return dist, door_open

# ===============


def image_resizing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(gray, (96, 96))

    return image

# ====================


def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding
# ========================


threshold = 0.65
interval = 0.3


def confidence_value(ref_encode, img_encode, thres=threshold):
    dist = np.linalg.norm((img_encode-ref_encode))
    confidence = (threshold-max([dist, interval]))/(threshold-interval)
    return dist, confidence

# # =================


# @app.route('/')
# def home():
#     return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    img1 = request.files['files'].read()
    img2 = request.files['file'].read()

    npimg1 = np.fromstring(img1, np.uint8)
    img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)

    npimg2 = np.fromstring(img2, np.uint8)
    img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)

    i1 = image_resizing(img1)
    i2 = image_resizing(img2)
    enc1 = img_to_encoding(i1, triplet_model)
    enc2 = img_to_encoding(i2, triplet_model)
    dist, conf = confidence_value(enc1, enc2)
    if dist < threshold:
        return render_template('index.html', Submit='Match')
    else:
        return render_template('index.html', Submit='No Match')


if __name__ == '__main__':
    app.run(debug=True)
