# -*- codeing = utf-8 -*-
# 导入必要的库
import random
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import GlobalMaxPooling1D
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Add
from keras.layers import Conv1D, LSTM, Bidirectional, Dense, BatchNormalization, GRU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import custom_object_scope
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
# 清空当前的TensorFlow会话
K.clear_session()
# 设置Python随机种子
random.seed(42)
# 设置NumPy随机种子
np.random.seed(42)
# 设置TensorFlow随机种子
tf.random.set_seed(42)

# 忽略警告
warnings.filterwarnings('ignore')

# 从CSV文件中读取数据
df = pd.read_csv("./10000jiazaosheng.csv", encoding='gb18030')
data = df.values
print("数据集维度：", data.shape)

# 定义滑动窗口大小
steps = 10

# 初始化输入（inp）和输出（out1, out2, out3）列表
inp = []
out1 = []
out2 = []
out3 = []

# 使用滑动窗口构建输入和输出序列
for i in range(len(data) - steps):
    inp.append(data[i:i + steps, 0:5])
    out1.append(data[i + steps, 5:6])
    out2.append(data[i + steps, 6:7])
    out3.append(data[i + steps, 7:8])

# 将列表转换为NumPy数组
inp = np.asarray(inp)
out1 = np.asarray(out1)
out2 = np.asarray(out2)
out3 = np.asarray(out3)

print("滑动窗口为5的数据集维度：", inp.shape)

# 使用MinMaxScaler进行输出特征缩放
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
out1 = scaler1.fit_transform(out1)
out2 = scaler2.fit_transform(out2)
out3 = scaler3.fit_transform(out3)

# 定义训练集和测试集的长度
train_length = 3000
test_length = 1000

# 定义训练过程中的一些超参数
patience = 10
epochs = 50
model_branch11_ratio = 0.7
model_branch12_ratio = 0.3
model_branch21_ratio = 0.7
model_branch22_ratio = 0.3
model_branch31_ratio = 0.7
model_branch32_ratio = 0.3
model_branch1_ratio = tf.Variable(initial_value=0.5, trainable=True)
model_branch2_ratio = tf.Variable(initial_value=0.3, trainable=True)
model_branch3_ratio = tf.Variable(initial_value=0.5, trainable=True)

# 划分训练集和测试集
x_train = inp[:train_length, :, :]
x_test = inp[train_length:train_length + test_length, :, :]
y_train1 = out1[:train_length]
y_test1 = out1[train_length:train_length + test_length]
y_train2 = out2[:train_length]
y_test2 = out2[train_length:train_length + test_length]
y_train3 = out3[:train_length]
y_test3 = out3[train_length:train_length + test_length]

def sliding_window_split(x_train, y_train, window_size, overlap_ratio, num_folds):
    # 计算滑动窗口的步长
    stride = int(val_steps * (1 - overlap_ratio))

    x_folds = []
    y_folds = []
    # 划分交叉验证集的每个折叠
    for i in range(num_folds):
        start = i * stride
        end = start + val_steps

        # 从训练集中提取滑动窗口数据
        x_fold = x_train[start:end]
        y_fold = y_train[start:end]

        x_folds.append(x_fold)
        y_folds.append(y_fold)

    return np.array(x_folds), np.array(y_folds)

# 定义参数
window_size = steps
val_steps = 5
overlap_ratio = 0.5
num_folds = 5

# 划分滑动窗口的交叉验证集
x_train_folds, y_train_folds1 = sliding_window_split(x_train, y_train1, window_size, overlap_ratio, num_folds)
print(x_train_folds.shape)
print(y_train_folds1.shape)

# 调整交叉验证集的维度与训练集相同
x_train_folds = np.reshape(x_train_folds, (-1, window_size, x_train.shape[2]))
y_train_folds1 = np.reshape(y_train_folds1, (-1, y_train1.shape[1]))
print(x_train_folds.shape)
print(y_train_folds1.shape)

# 打印结果
print("训练集维度:", x_train.shape, y_train1.shape)
print("交叉验证集1维度:", x_train_folds.shape, y_train_folds1.shape)

# 划分滑动窗口的交叉验证集（目标是y_train2）
x_train_folds, y_train_folds2 = sliding_window_split(x_train, y_train2, window_size, overlap_ratio, num_folds)

# 调整交叉验证集的维度与训练集相同
x_train_folds = np.reshape(x_train_folds, (-1, window_size, x_train.shape[2]))
y_train_folds2 = np.reshape(y_train_folds2, (-1, y_train2.shape[1]))


print("交叉验证集2维度:", x_train_folds.shape, y_train_folds2.shape)

# 划分滑动窗口的交叉验证集（目标是y_train3）
x_train_folds, y_train_folds3 = sliding_window_split(x_train, y_train3, window_size, overlap_ratio, num_folds)

# 调整交叉验证集的维度与训练集相同
x_train_folds = np.reshape(x_train_folds, (-1, window_size, x_train.shape[2]))
y_train_folds3 = np.reshape(y_train_folds3, (-1, y_train3.shape[1]))

print("交叉验证集3维度:", x_train_folds.shape, y_train_folds3.shape)


# ##########################################################并行网络##################################################
# 定义输入层
input_1 = Input(shape=(steps, 5), name='input_1')

# 对输入数据进行批量归一化
model_root = BatchNormalization()(input_1)

# 卷积层
model_root = Conv1D(filters=128, kernel_size=2, activation='relu', padding='same')(model_root)

# Dropout层，用于防止过拟合
model_root = Dropout(0.1)(model_root)

# 注意力层
model_root = SeqSelfAttention(attention_activation='relu')(model_root)

# 第一个分支，使用双向LSTM
model_branch11 = Bidirectional(LSTM(96, activation='relu', return_sequences=True))(model_root)
model_branch11 = Dropout(0.1)(model_branch11)
model_branch11 = SeqSelfAttention(attention_activation='relu')(model_branch11)
model_branch11 = LSTM(units=128, return_sequences=True, activation='relu')(model_branch11)
model_branch11 = SeqSelfAttention(128, attention_activation='relu')(model_branch11)
model_branch11 = Dropout(0.1)(model_branch11)
model_branch11 = GlobalMaxPooling1D()(model_branch11)

# 第二个分支，使用GRU
model_branch12 = GRU(units=96, return_sequences=True, activation='relu')(model_root)
model_branch12 = Dropout(0.1)(model_branch12)
model_branch12 = SeqSelfAttention(attention_activation='relu')(model_branch12)
model_branch12 = GRU(units=128, return_sequences=True, activation='relu')(model_branch12)
model_branch12 = SeqSelfAttention(128, attention_activation='relu')(model_branch12)
model_branch12 = Dropout(0.1)(model_branch12)
model_branch12 = GlobalMaxPooling1D()(model_branch12)

# 对两个分支的输出进行加权求和
x1 = Add()([model_branch11_ratio * model_branch11, model_branch12_ratio * model_branch12])

# Dropout层，用于防止过拟合
x1 = Dropout(0.2)(x1)

# 全连接层
x1 = Dense(100)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(10)(x1)
x1 = Dropout(0.1)(x1)

# 输出层，输出一个值
x1 = Dense(1)(x1)

# 构建模型
model_bingxing = Model(inputs=[input_1], outputs=[x1])

print('并行网络训练中！！')

# 编译模型
model_bingxing.compile(loss='mse', optimizer='adam')

# 定义模型检查点，保存在验证集上表现最好的模型
checkpoint_bingxing1 = ModelCheckpoint("best_model_bingxing1.hdf5", monitor="val_loss", mode="min", save_best_only=True)

# 训练第一个目标（y_train1）
history_bingxing1 = model_bingxing.fit(x_train, y_train1, epochs=epochs, verbose=1, callbacks=[checkpoint_bingxing1], validation_data=(x_train_folds, y_train_folds1))

# 定义模型检查点，保存在验证集上表现最好的模型
checkpoint_bingxing2 = ModelCheckpoint("best_model_bingxing2.hdf5", monitor="val_loss", mode="min", save_best_only=True)

# 训练第二个目标（y_train2）
history_bingxing2 = model_bingxing.fit(x_train, y_train2, epochs=epochs, verbose=1, callbacks=[checkpoint_bingxing2], validation_data=(x_train_folds, y_train_folds2))

# 定义模型检查点，保存在验证集上表现最好的模型
checkpoint_bingxing3 = ModelCheckpoint("best_model_bingxing3.hdf5", monitor="val_loss", mode="min", save_best_only=True)

# 训练第三个目标（y_train3）
history_bingxing3 = model_bingxing.fit(x_train, y_train3, epochs=epochs, verbose=1, callbacks=[checkpoint_bingxing3], validation_data=(x_train_folds, y_train_folds3))

print('并行网络训练完成！！')
#####################################################################################################################################################
# 设置SeqSelfAttention为自定义对象，以便正确加载模型
with custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
    # 加载第一个目标的模型
    model_bingxing1 = tf.keras.models.load_model('best_model_bingxing1.hdf5', options={'encoding': 'gbk'})

    # 加载第二个目标的模型
    model_bingxing2 = tf.keras.models.load_model('best_model_bingxing2.hdf5', options={'encoding': 'gbk'})

    # 加载第三个目标的模型
    model_bingxing3 = tf.keras.models.load_model('best_model_bingxing3.hdf5', options={'encoding': 'gbk'})

# 使用加载的模型进行预测
predict_bingxing1 = model_bingxing1.predict(x_test)
predict_bingxing2 = model_bingxing2.predict(x_test)
predict_bingxing3 = model_bingxing3.predict(x_test)



#####################################################CNN+LSTM############################################################
# # 定义 CNN+LSTM 模型
model_cnn_lstm = Sequential()
model_cnn_lstm.add(BatchNormalization())  # 添加归一化层
model_cnn_lstm.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
model_cnn_lstm.add(LSTM(units=100, return_sequences=True, activation='relu'))
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=False))
model_cnn_lstm.add(Dense(100, activation='relu'))
model_cnn_lstm.add(Dense(1))

# 编译 CNN+LSTM 模型
model_cnn_lstm.compile(loss='mse', optimizer='adam')

print('CNN+LSTM 网络训练中！！')

# 设置早停和模型检查点
early_stop_cnn_lstm1 = EarlyStopping(monitor="loss", mode="min", patience=patience)
checkpoint_cnn_lstm1 = ModelCheckpoint("best_model_cnn_lstm1.hdf5", monitor="val_loss", mode="min", save_best_only=True)

#训练 CNN+LSTM 模型
history_cnn_lstm1 = model_cnn_lstm.fit(x_train, y_train1, epochs=epochs, verbose=1, callbacks=[checkpoint_cnn_lstm1], validation_data=(x_train_folds, y_train_folds1))

checkpoint_cnn_lstm2 = ModelCheckpoint("best_model_cnn_lstm2.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_cnn_lstm2 = model_cnn_lstm.fit(x_train, y_train2, epochs=epochs, verbose=1, callbacks=[checkpoint_cnn_lstm2], validation_data=(x_train_folds, y_train_folds2))

checkpoint_cnn_lstm3 = ModelCheckpoint("best_model_cnn_lstm3.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_cnn_lstm3 = model_cnn_lstm.fit(x_train, y_train3, epochs=epochs, verbose=1, callbacks=[checkpoint_cnn_lstm3], validation_data=(x_train_folds, y_train_folds3))

print('CNN+LSTM 网络训练完成！！')
##############################################################################################################################################
# 加载 CNN+LSTM 模型
with custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
    model_cnn_lstm1 = tf.keras.models.load_model('best_model_cnn_lstm1.hdf5', options={'encoding': 'gbk'})
    model_cnn_lstm2 = tf.keras.models.load_model('best_model_cnn_lstm2.hdf5', options={'encoding': 'gbk'})
    model_cnn_lstm3 = tf.keras.models.load_model('best_model_cnn_lstm3.hdf5', options={'encoding': 'gbk'})

# 对测试集进行预测
predict_cnn_lstm1 = model_cnn_lstm1.predict(x_test)
predict_cnn_lstm2 = model_cnn_lstm2.predict(x_test)
predict_cnn_lstm3 = model_cnn_lstm3.predict(x_test)


###############################################LSTM#####################################################################
# 定义 LSTM 模型
model_lstm = Sequential()
model_lstm.add(BatchNormalization())  # 添加归一化层
model_lstm.add(LSTM(units=128, return_sequences=True, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=128, return_sequences=True, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(128, return_sequences=True, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=128, return_sequences=True, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(128, return_sequences=False, activation='relu'))
model_lstm.add(Flatten())
model_lstm.add(Dense(1))

# 编译 LSTM 模型
model_lstm.compile(loss='mse', optimizer='adam')

print('LSTM 网络训练中！！')

# 设置早停和模型检查点
early_stop_lstm1 = EarlyStopping(monitor="loss", mode="min", patience=patience)
checkpoint_lstm1 = ModelCheckpoint("best_model_lstm1.hdf5", monitor="val_loss", mode="min", save_best_only=True)

# 训练 LSTM 模型
history_lstm1 = model_lstm.fit(x_train, y_train1, epochs=epochs, verbose=1, callbacks=[checkpoint_lstm1], validation_data=(x_train_folds, y_train_folds1))

checkpoint_lstm2 = ModelCheckpoint("best_model_lstm2.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_lstm2 = model_lstm.fit(x_train, y_train2, epochs=epochs, verbose=1, callbacks=[checkpoint_lstm2], validation_data=(x_train_folds, y_train_folds2))

checkpoint_lstm3 = ModelCheckpoint("best_model_lstm3.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_lstm3 = model_lstm.fit(x_train, y_train3, epochs=epochs, verbose=1, callbacks=[checkpoint_lstm3], validation_data=(x_train_folds, y_train_folds3))

print('LSTM 网络训练完成！！')
####################################################################################################################################
# 加载 LSTM 模型
with custom_object_scope({'LSTM': LSTM}):
    model_lstm1 = tf.keras.models.load_model('best_model_lstm1.hdf5', options={'encoding': 'gbk'})
    model_lstm2 = tf.keras.models.load_model('best_model_lstm2.hdf5', options={'encoding': 'gbk'})
    model_lstm3 = tf.keras.models.load_model('best_model_lstm3.hdf5', options={'encoding': 'gbk'})

# 对测试集进行预测
predict_lstm1 = model_lstm1.predict(x_test)
predict_lstm2 = model_lstm2.predict(x_test)
predict_lstm3 = model_lstm3.predict(x_test)

#################################################### GRU ##########################################################################

# 定义 GRU 模型
model_gru = Sequential()
model_gru.add(BatchNormalization())  # 添加归一化层
model_gru.add(GRU(units=128, return_sequences=True, activation='relu'))
model_gru.add(Dropout(0.2))
model_gru.add(GRU(units=128, return_sequences=True, activation='relu'))
model_gru.add(Dropout(0.1))
model_gru.add(GRU(units=128, return_sequences=True, activation='relu'))
model_gru.add(Dropout(0.2))
model_gru.add(GRU(units=128, return_sequences=True, activation='relu'))
model_gru.add(Dropout(0.1))
model_gru.add(GRU(units=128, return_sequences=True, activation='relu'))
model_gru.add(Dropout(0.2))
model_gru.add(Flatten())
model_gru.add(Dense(1))

# 编译 GRU 模型
model_gru.compile(loss='mse', optimizer='adam')

print('GRU 网络训练中！！')

# 设置早停和模型检查点
early_stop_gru1 = EarlyStopping(monitor="loss", mode="min", patience=patience)
checkpoint_gru1 = ModelCheckpoint("best_model_gru1.hdf5", monitor="val_loss", mode="min", save_best_only=True)

# 训练 GRU 模型
history_gru1 = model_gru.fit(x_train, y_train1, epochs=epochs, verbose=1, callbacks=[checkpoint_gru1], validation_data=(x_train_folds, y_train_folds1))

checkpoint_gru2 = ModelCheckpoint("best_model_gru2.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_gru2 = model_gru.fit(x_train, y_train2, epochs=epochs, verbose=1, callbacks=[checkpoint_gru2], validation_data=(x_train_folds, y_train_folds2))

checkpoint_gru3 = ModelCheckpoint("best_model_gru3.hdf5", monitor="val_loss", mode="min", save_best_only=True)
history_gru3 = model_gru.fit(x_train, y_train3, epochs=epochs, verbose=1, callbacks=[checkpoint_gru3], validation_data=(x_train_folds, y_train_folds3))

print('GRU 网络训练完成！！')
########################################################################################################3
# 加载 GRU 模型
with custom_object_scope({'GRU': GRU}):
    model_gru1 = tf.keras.models.load_model('best_model_gru1.hdf5', options={'encoding': 'gbk'})
    model_gru2 = tf.keras.models.load_model('best_model_gru2.hdf5', options={'encoding': 'gbk'})
    model_gru3 = tf.keras.models.load_model('best_model_gru3.hdf5', options={'encoding': 'gbk'})

# 对测试集进行预测
predict_gru1 = model_gru1.predict(x_test)
predict_gru2 = model_gru2.predict(x_test)
predict_gru3 = model_gru3.predict(x_test)
##################################################反归一化##############################################################
# 反归一化预测结果
y_test1 = scaler1.inverse_transform(y_test1)
y_test2 = scaler2.inverse_transform(y_test2)
y_test3 = scaler3.inverse_transform(y_test3)

predict_bingxing1 = scaler1.inverse_transform(predict_bingxing1)
predict_bingxing2 = scaler2.inverse_transform(predict_bingxing2)
predict_bingxing3 = scaler3.inverse_transform(predict_bingxing3)

predict_cnn_lstm1 = scaler1.inverse_transform((predict_cnn_lstm1))
predict_cnn_lstm2 = scaler2.inverse_transform((predict_cnn_lstm2))
predict_cnn_lstm3 = scaler3.inverse_transform((predict_cnn_lstm3))

predict_lstm1 = scaler1.inverse_transform(predict_lstm1)
predict_lstm2 = scaler2.inverse_transform(predict_lstm2)
predict_lstm3 = scaler3.inverse_transform(predict_lstm3)

predict_gru1 = scaler1.inverse_transform(predict_gru1)
predict_gru2 = scaler2.inverse_transform(predict_gru2)
predict_gru3 = scaler3.inverse_transform(predict_gru3)

####################################################打印数据###########################################################

print('并行网络预测Y1的均方误差：', mean_squared_error(y_test1, predict_bingxing1))
print('CNN+LSTM网络预测Y1的均方误差：', mean_squared_error(y_test1, predict_cnn_lstm1))
print('LSTM网络预测Y1的均方误差：', mean_squared_error(y_test1, predict_lstm1))
print('GRU网络预测Y1的均方误差：', mean_squared_error(y_test1, predict_gru1))

print('并行网络预测Y2的均方误差：', mean_squared_error(y_test2, predict_bingxing2))
print('CNN+LSTM网络预测Y2的均方误差：', mean_squared_error(y_test2, predict_cnn_lstm2))
print('LSTM网络预测Y2的均方误差：', mean_squared_error(y_test2, predict_lstm2))
print('GRU网络预测Y2的均方误差：', mean_squared_error(y_test2, predict_gru2))

print('并行网络预测Y3的均方误差：', mean_squared_error(y_test3, predict_bingxing3))
print('CNN+LSTM网络预测Y3的均方误差：', mean_squared_error(y_test3, predict_cnn_lstm3))
print('LSTM网络预测Y3的均方误差：', mean_squared_error(y_test3, predict_lstm3))
print('GRU网络预测Y3的均方误差：', mean_squared_error(y_test3, predict_gru3))

# 构造保存结果的字典
Result = {'y_test1': y_test1, 'predict_bingxing1': predict_bingxing1, 'predict_cnn_lstm1': predict_cnn_lstm1, 'predict_lstm1': predict_lstm1, 'predict_gru1': predict_gru1,
        'y_test2': y_test2, 'predict_bingxing2': predict_bingxing2, 'predict_lstm2': predict_lstm2, 'predict_cnn_lstm2': predict_cnn_lstm2, 'predict_gru2': predict_gru2,
        'y_test3': y_test3, 'predict_bingxing3': predict_bingxing3, 'predict_cnn_lstm3': predict_cnn_lstm3, 'predict_lstm3': predict_lstm3, 'predict_gru3': predict_gru3}

# 保存结果为MAT文件
sio.savemat('result.mat', Result)

# 构造保存损失记录的字典
# loss_result = {'bingxing_loss1':history_bingxing1.history['loss'],'bingxing_loss2':history_bingxing2.history['loss'],'bingxing_loss3':history_bingxing3.history['loss'],
#                'cnn_lstm_loss1':history_cnn_lstm1.history['loss'],'cnn_lstm_loss2':history_cnn_lstm2.history['loss'],'cnn_lstm_loss3':history_cnn_lstm3.history['loss'],
#                'lstm_loss1':history_lstm1.history['loss'],'lstm_loss2':history_lstm2.history['loss'],'lstm_loss3':history_lstm3.history['loss'],
#                'gru_loss1':history_gru1.history['loss'],'gru_loss2':history_gru2.history['loss'],'gru_loss3':history_gru3.history['loss']}
#
# # 保存损失记录为MAT文件
# sio.savemat('loss.mat', loss_result)

# 构造保存验证集损失记录的字典
# val_loss_result = {'bingxing_val_loss1':history_bingxing1.history['val_loss'],'bingxing_val_loss2':history_bingxing2.history['val_loss'],'bingxing_val_loss3':history_bingxing3.history['val_loss'],
#                'cnn_lstm_val_loss1':history_cnn_lstm1.history['val_loss'],'cnn_lstm_val_loss2':history_cnn_lstm2.history['val_loss'],'cnn_lstm_val_loss3':history_cnn_lstm3.history['val_loss'],
#                'lstm_val_loss1':history_lstm1.history['val_loss'],'lstm_val_loss2':history_lstm2.history['val_loss'],'lstm_val_loss3':history_lstm3.history['val_loss'],
#                'gru_val_loss1':history_gru1.history['val_loss'],'gru_val_loss2':history_gru2.history['val_loss'],'gru_val_loss3':history_gru3.history['val_loss']}
#
# # 保存验证集损失记录为MAT文件
# sio.savemat('val_loss.mat', val_loss_result)

##################################################画图#################################################################
# 绘制 Y1 的测试结果和各模型的预测结果
plt.figure(1, figsize=(32, 8))  # 设置图形大小为原来的两倍
plt.plot(y_test1, linewidth=1.5)  # 实际值

plt.plot(predict_cnn_lstm1.reshape(-1,), linewidth=1)  # CNN+LSTM网络预测结果
plt.plot(predict_lstm1.reshape(-1,), linewidth=1)  # LSTM网络预测结果
plt.plot(predict_gru1.reshape(-1,), linewidth=1)  # GRU网络预测结果
plt.plot(predict_bingxing1.reshape(-1,), linewidth=1)  # 并行网络预测结果
plt.legend(('Test', 'Predicted(bingxing)', 'Predicted(cnn+lstm)', 'Predicted(lstm)', 'Predicted(gru)'))
plt.title('Y1')
plt.savefig('output_plot1.png')
plt.show()

# 绘制 Y2 的测试结果和各模型的预测结果
plt.figure(2, figsize=(32, 8))  # 设置图形大小为原来的两倍
plt.plot(y_test2, linewidth=1.5)  # 实际值

plt.plot(predict_cnn_lstm2.reshape(-1,), linewidth=1)  # CNN+LSTM网络预测结果
plt.plot(predict_lstm2.reshape(-1,), linewidth=1)  # LSTM网络预测结果
plt.plot(predict_gru2.reshape(-1,), linewidth=1)  # GRU网络预测结果
plt.plot(predict_bingxing2.reshape(-1,), linewidth=1)  # 并行网络预测结果
plt.legend(('Test', 'Predicted(bingxing)', 'Predicted(cnn+lstm)', 'Predicted(lstm)', 'Predicted(gru)'))
plt.title('Y2')
plt.savefig('output_plot2.png')
plt.show()

# 绘制 Y3 的测试结果和各模型的预测结果
plt.figure(3, figsize=(32, 8))  # 设置图形大小为原来的两倍
plt.plot(y_test3, linewidth=1.5)  # 实际值

plt.plot(predict_cnn_lstm3.reshape(-1,), linewidth=1)  # CNN+LSTM网络预测结果
plt.plot(predict_lstm3.reshape(-1,), linewidth=1)  # LSTM网络预测结果
plt.plot(predict_gru3.reshape(-1,), linewidth=1)  # GRU网络预测结果
plt.plot(predict_bingxing3.reshape(-1,), linewidth=1)  # 并行网络预测结果
plt.legend(('Test', 'Predicted(bingxing)', 'Predicted(cnn+lstm)', 'Predicted(lstm)', 'Predicted(gru)'))
plt.title('Y3')

# 保存图形
plt.savefig('output_plot3.png')

# 显示图形
plt.show()