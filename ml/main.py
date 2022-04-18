import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import Image
import math

IMG_W, IMG_H = 320, 240
LOAD = True
FIT = True

def load_data(path='./data/'):
    print('loading from', path, path + "/*.txt")
    dataset = tf.keras.utils.image_dataset_from_directory(path, labels=None, shuffle=False, color_mode='rgb',
                                                          batch_size=None, image_size=(IMG_H, IMG_W))
    lables = tf.data.Dataset.list_files(path + "/*.txt", shuffle=False)
    lables = lables.map(lambda file_path: tf.strings.to_number(
        tf.strings.split(tf.strings.split(tf.io.read_file(file_path), '\n')[0], ',')))
    print('lens', len(lables), len(dataset))
    return np.array(list(dataset)), np.array(list(lables))


def filter_out_zeroes(l):
    print('filtering')
    # lx, ly = l
    # print(l)
    filter_cond = np.logical_and.reduce(l[1] > 0, 1)
    return l[0][filter_cond], l[1][filter_cond]


def normolize(l):
    print('norm')
    # lx, ly = l
    coef = np.array([IMG_H, IMG_W]) / 2
    return l[0] / 255, (l[1] - coef) / coef


def denorm(y):
    print('denorm')
    coef = np.array([IMG_H, IMG_W]) / 2
    return np.ceil(y * coef + coef)


def shuffle(l, seed=0):
    print('shuffling')
    # lx, ly = l
    old_seed = random.randint(0, np.power(2, 32))
    random.seed(seed)
    li = list(range(l[0].shape[0]))
    random.shuffle(li)
    random.seed(old_seed)
    li = np.array(li)
    return l[0][li], l[1][li]


def split_train_validate(l, split=0.05, seed=0):
    print('splitting')
    # lx, ly = l
    old_seed = random.randint(0, np.power(2, 32))
    random.seed(seed)
    sample = np.array(random.sample(range(l[0].shape[0]), math.ceil(l[0].shape[0] * split)))
    random.seed(old_seed)
    tr = np.full(l[0].shape[0], True)
    tr[sample] = False
    return l[0][tr], l[1][tr], l[0][np.logical_not(tr)], l[1][np.logical_not(tr)]


def load_and_prepare_data(data_path):
    print('lapd')
    return split_train_validate(shuffle(normolize(filter_out_zeroes(load_data(data_path)))))


def create_model():
    print('creating model')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMG_H, IMG_W)),
        tf.keras.layers.Dense(3),
        tf.keras.layers.Dense(2)
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()
    return model


def load_model(save_path="training_1/cp.ckpt"):
    print('loading model')
    model = create_model()
    model.load_weights(save_path)
    return model


def fit_model(model, lx, ly, save_path="training_1/cp.ckpt"):
    print('fitting model')
    checkpoint_path = save_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(lx, ly, epochs=50, callbacks=[cp_callback])


def to_greyscale(x):
    print('to greyscale')
    coef = np.array([0.299, 0.587, 0.114])
    return np.sum(x * coef, axis=3)


def predicted_stats(sx, vy, predicted, save_prefix=''):
    print('Evaluated\n', predicted)
    print('Expected\n', vy)
    print('Diff\n', predicted - vy)
    print('Second norm error\n', np.sum(np.sqrt(predicted * predicted + vy * vy), 1))
    print('Mean squared error', np.mean(np.sum(np.sqrt(predicted * predicted + vy * vy), 1)))

    dvy = denorm(vy).astype(np.int32)
    dpr = denorm(predicted).astype(np.int32)
    print(dpr)
    sx = (sx * 255).astype(np.uint8)
    for i, (el, dvyi, dpri) in enumerate(zip(sx, dvy, dpr)):
        # el = np.transpose(el, axes=(1, 0, 2))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                x = dvyi[0] + di
                y = dvyi[1] + dj
                if x < 0 or x >= IMG_W:
                    print('ERROR: x out of bounds for dvui', dvyi, 'for image ', i)
                    continue
                if y < 0 or y >= IMG_H:
                    print('ERROR: x out of bounds for dvui', dvyi, 'for image ', i)
                    continue
                el[y, x] = np.array([255, 0, 0])

                x = dpri[0] + di
                y = dpri[1] + dj
                if x < 0 or x >= IMG_W:
                    print('ERROR: x out of bounds for dpri', dpri, 'for image ', i)
                    continue
                if y < 0 or y >= IMG_H:
                    print('ERROR: x out of bounds for dpri', dpri, 'for image ', i)
                    continue
                el[y, x] = np.array([0, 255, 0])

        Image.fromarray(el).save('./predicted/' + save_prefix + str(i) + '.png')
        # plt.imshow(el, interpolation='nearest')
        # plt.savefig('./predicted/' + str(i) + '.png')

def check_vis(sx, vy):
    dvy = denorm(vy).astype(np.int32)
    sx = (sx * 255).astype(np.uint8)
    for i, (el, dvyi) in enumerate(zip(sx, dvy)):
        # el = np.transpose(el, axes=(1, 0, 2))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                x = dvyi[0] + di
                y = dvyi[1] + dj
                if x < 0 or x > IMG_W:
                    print('ERROR: x out of bounds for dvui', dvyi, 'for image ', i)
                    continue
                if y < 0 or y > IMG_H:
                    print('ERROR: x out of bounds for dvui', dvyi, 'for image ', i)
                    continue
                el[y, x] = np.array([255, 0, 0])
        
        Image.fromarray(el).save('./test/' + save_prefix + str(i) + '.png')
        # plt.imshow(el, interpolation='nearest')
        # plt.savefig('./test/' + str(i) + '.png')

def main(data_path='./data', save_prefix=''):
    lx, ly, sx, vy = load_and_prepare_data(data_path)
    # check_vis(lx, ly)
    lx = to_greyscale(lx)
    vx = to_greyscale(sx)
    print(lx[:, 0, 0])
    print(ly)
    print('-------------')
    print(vx[:, 0, 0])
    print(vy)

    if LOAD:
        model = load_model()
    else:
        model = create_model()
    if FIT:
        fit_model(model, lx, ly)


    model.evaluate(lx, ly, verbose=2)
    model.evaluate(vx, vy, verbose=2)
    print(sx, vy)
    predicted_stats(sx, vy, model(vx).numpy(), save_prefix=save_prefix)


if __name__ == '__main__':
    LOAD = True
    for folder in os.listdir('./data'):
        main(data_path='./data/' + folder, save_prefix=folder + '_')
        LOAD = True
