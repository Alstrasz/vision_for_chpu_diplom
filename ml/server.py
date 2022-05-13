from flask import Flask
from flask import request
import json
import numpy as np
from PIL import Image
from shape_gen import ShapeGen
from main import load_model
from main import mark_dot
from main import denorm

app = Flask(__name__)
shape = ShapeGen()
model = load_model("training_1/cp.ckpt")
saved_count = 0

@app.route("/")
def index():
    return "Http server to access nn for predicting edge. Use get/post/delete endpoints at /nn"


@app.route('/nn', methods=['POST'])
def nn_endpoint():
    global model, saved_count
    print(request.method)
    if request.method == 'POST':
        body = json.loads(request.data)
        image = np.array(body['image'])
        sx = (image * 255).astype(np.uint8).reshape((image.shape[0], image.shape[1], 1)).repeat(3, axis=2)

        Image.fromarray(sx).save('./server/' + 'test.png')
        reshaped_image = image.reshape((1, image.shape[0], image.shape[1]))
        predicted = model(reshaped_image).numpy()[0]
        # predicted = np.array([0.5, 0])
        print(denorm(predicted))
        mark_dot(sx, denorm(predicted).astype(np.int32), np.array([0, 255, 0]))
        save_suffix = '_' + str(saved_count)
        if 'save_suffix' in body:
            save_suffix = body['save_suffix']
        Image.fromarray(sx).save('./server/' + 'predicted' + save_suffix + '.png')
        if 'scale' in body:
            predicted[0] *= body['scale'][0]
            predicted[1] *= body['scale'][1]
        if 'dot' in body and 'key' in body:
            dot = body['dot']
            key = body['key']
            shape.add_dot(key, dot)
            shape.get_image(key)
            if shape.is_complete(key):
                predicted[0] = 0
                predicted[1] = 0
        return {
            'predicted': predicted.tolist()
        }


@app.route('/shape', methods=['GET', 'PUT', 'DELETE'])
def shape_endpoint():
    global shape
    body = json.loads(request.data)
    key = body['key']
    if request.method == 'GET':
        return {
            'dots': shape.get_dots(key)
        }
    if request.method == 'PUT':
        dot = body['dot']
        shape.add_dot(key, dot)
        shape.get_image(key)
    if request.method == 'DELETE':
        shape.remove_dots(key)
    return {
        'status': 200
    }


if __name__ == '__main__':
    app.run(port=8000, host='127.0.0.1')
