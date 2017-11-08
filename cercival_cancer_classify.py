import numpy as np
import pandas as pd
import csv
from subprocess import check_output
print(check_output(['ls','input']).decode('utf8'))
#np.set_printoptions(threshold=np.inf)   #全部输出
import glob
from PIL import ImageFilter, ImageStat, Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import cv2
from sklearn.model_selection import KFold
#Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，
# 如果池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，
# 直到池中有进程结束，才会创建新的进程来执行这些请求。
#multiprocessing包是Python中的多进程管理包,可以利用multiprocessing.Process对象来创建一个进程
from multiprocessing import Pool, cpu_count
#并发的线程数,在multiprocessing模块中的cpu_count()函数已经实现了该功能
import yaml
train = glob.glob("input/train/*/*.jpg")
#print(len(train))
train = pd.DataFrame([[p.split('/')[2],p.split('/')[3],p] for p in train], columns = ['type','image','path'])
#print(train)




test = glob.glob("input/test/*.jpg")
#print(test)
test = pd.DataFrame([[p.split('/')[2],p] for p in test], columns = ['image','path'])
#print(test)
test_id = test.image.values

'''
types = train.groupby('type', as_index=False).count()
types.plot(kind='bar', x='type', y='path', figsize=(7,4))
#plt.show()
#Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，
# 如果池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，
# 直到池中有进程结束，才会创建新的进程来执行这些请求。
#multiprocessing包是Python中的多进程管理包,可以利用multiprocessing.Process对象来创建一个进程
#并发的线程数,在multiprocessing模块中的cpu_count()函数已经实现了该功能
def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

#train=im_stats_df
def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    #从train里提取出路径传递给im_multi函数，函数执行后返回路径和图片的大小并且赋值给ret=[path, {'size': [0,0]}]
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        #('input/train/Type_2/946.jpg', {'size': (2448, 3264)})
        im_stats_d[ret[i][0]] = ret[i][1]

    #print(im_stats_d.items())
    #按照X取出size并放入 im_stats_df中
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    #print(im_stats_df)
    return im_stats_df
#对图片的大小进行改变
def get_im_cv2(path):
    img = cv2.imread(path)
    #线性插值法
    resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR) #use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, resized]
#对图片数据进行归一化操作
def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    #print(ret)
    #['input/train/Type_2/1218.jpg', array([[[1, 1, 1],.....[0, 0, 0]]], dtype=uint8)]
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    #list转成dict
    #print(imf_d)
    ret = []
    #imf_d.shape={'input/train/Type_2/1218.jpg' : array([[[1, 1, 1],.....[0, 0, 0]]], dtype=uint8)}
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    #(1481, 32, 32, 3)
    #print(fdata.shape)
    #矩阵的转置
    fdata = fdata.transpose((0, 3, 1, 2))
    #(1481, 3, 32, 32),3代表--->RGB3通道
    #print(fdata.shape)
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata
#limit for Kaggle Demo
train = im_stats(train)
#print(train)
#去除size为0的图片
train = train[train['size'] != '0 0'].reset_index(drop=True) #corrupt images removed
print("Bad images removed")
print("loading train data")
#数据归一化
train_data = normalize_image_features(train['path'])
print("train data saved")
np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)


from sklearn.preprocessing import LabelEncoder
#LabelEncoder 是对不连续的数字或者文本进行编号
#([1,1,100,67,5])
#array([0,0,3,2,1])
le = LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_)
print(train_target)
#['Type_1' 'Type_2' 'Type_3'],[1,0,2]
np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)


test_data = normalize_image_features(test['path'])
#(512, 3, 32, 32)
#print(test_data.shape)
np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)
print("test data loaded")
test_id = test.image.values
#print(test_id)#['67.jpg' '206.jpg' '357.jpg' '23.jpg'......
#从test中独处图片的序号
np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)
'''


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K,regularizers
K.set_image_dim_ordering('th')
K.set_floatx('float32')

import pandas as pd
import numpy as np
np.random.seed(17)

train_data = np.load('train.npy')
train_target = np.load('train_target.npy')
from keras.layers.normalization import  BatchNormalization
from keras.applications.vgg16 import VGG16
def create_model(opt_='nadam'):

#dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。
# 例如128*128的三通道彩色图片，在‘th’模式中input_shape应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），
# 注意这里3出现在第0个位置，因为input_shape不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。
# 默认是image_dim_ordering指定的模式，可在~/.keras/keras.json中查看，若没有设置过则为’tf’。
#sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.6, nesterov=True)
#sparse_categorical_crossentrop
#在上面的多分类的对数损失函数的基础上，增加了稀疏性（即数据中多包含一定0数据的数据集），如目录所说，需要对数据标签添加一个维度np.expand_dims(y,-1)。
#categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
#model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#filter=32,kernal=3x3 strides=1x1
#Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
    nb_classes = 3
    img_rows, img_cols = 128, 128
    img_channels = 3
    #init a model
    model = Sequential()
    #layer1:conv1
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    #layer2:conv2
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    #layer3:pool1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #layer4:conv3
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #layer5:conv4
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    #layer6:pool2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #layer7:connect1
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #softmax
    model.add(Dense(nb_classes,bias_regularizer=regularizers.l1(0.01)))
    model.add(Activation('softmax'))


    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Train...")
    # erbose：日志显示，0
    # 为不在标准输出流输出日志信息，1
    # 为输出进度条记录，2
    # 为每个epoch输出一行记录
    return model

#data augmentation
#keras.preprocessing.image.ImageDataGenerator()
#用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
#rotation_range：整数，数据提升时图片随机转动的角度.
#zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
def cleanImages():
    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    datagen.fit(train_data)
    return datagen


def fit():
    print("cleaning images")
    datagen = cleanImages()
    print("images cleaned")

    model = create_model()
    x_train, x_val_train, y_train, y_val_train = train_test_split(train_data, train_target, test_size=0.2,
                                                                  random_state=17)
    print("fitting data")
    #利用Python的生成器，逐个生成数据的batch并进行训练。
    # 生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
    #fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None,
    #  validation_data=None, validation_steps=None, class_weight=None,
    # max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=200, shuffle=True), nb_epoch=1000 ,
                        samples_per_epoch=len(x_train), verbose=1, validation_data=(x_val_train, y_val_train))
    print("Evaluate...")
   # score = model.evaluate(x_val_train,y_val_train,batch_size=15)
    yaml_string = model.to_yaml()
    with open('CNN.yml','w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))

    #summarize history for accuray
    plt.plot()
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('loss.png')

    accy = history.history['val_acc']
    np_accy = np.array(accy)
    np.savetxt('save_acc.txt',np_accy)




   #save accuray of text
    model.save_weights('CNN.h5')
#开始训练

fit()

def Predict():
   # model = fit()
    with open('CNN.yml','r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    model.load_weights('CNN.h5')

    print("data fitted in model")
    test_data = np.load('test.npy')
    test_id = np.load('test_id.npy')
    print("creating predictions")
    #model.compile()
    predictions = model.predict_proba(test_data)
    print("predictions made")
    return predictions


def createSub():
    pred = Predict()
    print("creating submission file")
    print(pred)
    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id
    df.to_csv('submission.csv', index=False)
    print("submission created")


if __name__ == '__main__':
    createSub()














