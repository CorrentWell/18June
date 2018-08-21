# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:41:51 2018

@author: coltn
"""

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.utils.np_utils import to_categorical
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# 分類するクラス
classes = ["BG", "HCC"]
nb_classes = len(classes)

img_width, img_height = 150, 150

# トレーニング用とバリデーション用の画像格納先
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# 今回はトレーニング用に200枚、バリデーション用に50枚の画像を用意した。
nb_train_samples = 880
nb_validation_samples = 220

batch_size = 20
nb_epoch = 300


result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def vgg_model_maker():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    return model

def random_crop(img, crop_size):
    if img.mode != "RGB":
        print("This image is not available.")
        return 0
    
    max_x, max_y = img.size
    crop_x, crop_y = crop_size
    px, py = random.randint(0, max_x-crop_x), random.randint(0, max_y-crop_y)
    img_crop = img.crop((px, py, px+crop_x, py+crop_y))
    img_array = np.asarray(img_crop)
    return img_array

def set_tensor():
    image_list = np.empty((0, img_width), float)
    label_list = []
    
    for d in os.listdir(train_data_dir):
        if d == ".DS_Store":
            continue
        
        d1 = train_data_dir + "/" + d
        label = 0
        
        if d == "BG":
            label = 0
        elif d == "HCC":
            label = 1
        
        for file in os.listdir(d1):
            if file != ".DS_Store":
                label_list.append(label)
                filepath = d1 + "/" + file
                image_list = np.append(image_list,
                                       np.asarray(random_crop(Image.open(filepath), (img_width, img_height)), dtype="float32") / 255.0,
                                       axis=0)
                #nom_image = (image / 255.0).tolist()
                #image_list.append(nom_image)
        
        image_list = image_list.reshape(-1, img_height, img_width, 3)
        Y = to_categorical(label_list)
        
    val_image_list = np.empty((0, img_width), float)
    val_label_list = []
    
    for d in os.listdir(validation_data_dir):
        if d == ".DS_Store":
            continue
        
        d1 = validation_data_dir + "/" + d
        label = 0
        
        if d == "BG":
            label = 0
        elif d == "HCC":
            label = 1
        
        for file in os.listdir(d1):
            if file != ".DS_Store":
                val_label_list.append(label)
                filepath = d1 + "/" + file
                val_image_list = np.append(val_image_list,
                                  np.asarray(random_crop(Image.open(filepath), (img_width, img_height)), dtype="float32") / 255.0,
                                  axis=0)
                #image = np.asarray(Image.open(filepath).resize((img_width, img_height), Image.LANCZOS))
                #nom_image = (image / 255.0).tolist()
                #val_image_list.append(nom_image)
        
        val_image_list = val_image_list.reshape(-1, img_height, img_width, 3)
        val_Y = to_categorical(val_label_list)
    
    return image_list, Y, val_image_list, val_Y

def image_generator():
    """ ディレクトリ内の画像を読み込んでトレーニングデータとバリデーションデータの作成 """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    return (train_generator, validation_generator)


if __name__ == '__main__':
    start = time.time()

    # モデル作成
    vgg_model = vgg_model_maker()
    
    
    # 最後のconv層の直前までの層をfreeze
    for layer in vgg_model.layers[:]:
        layer.trainable = True
    
    # 多クラス分類を指定
    vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

    # 画像のジェネレータ生成
    #train_generator, validation_generator = image_generator()
    
    x_train, y_train, x_test, y_test = set_tensor()

    # Fine-tuning
    history = vgg_model.fit(x_train, 
                            y_train, 
                            batch_size=batch_size, 
                            epochs=nb_epoch,
                            validation_data=(x_test, y_test), 
                            verbose=1, 
                            shuffle=True)

    vgg_model.save_weights(os.path.join(result_dir, 'finetuning.h5'))

    process_time = (time.time() - start) / 60
    print(u'学習終了。かかった時間は', process_time, u'分です。')
    plt.plot(range(1, nb_epoch+1), history.history['acc'], label="training")
    plt.plot(range(1, nb_epoch+1), history.history['val_acc'], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()