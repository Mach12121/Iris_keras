
# 一、 数据准备

本实验中使用的是Iris数据集，数据集中包含了三个不同种类的鸢尾花，每个类别包含50个样本，一共150个样本。数据集中的每个样本有4个特征，包括花萼长度、花萼宽度、花瓣长度和花瓣宽度。将数据集分为训练集和测试集，其中训练集占80%，测试集占20%。

以下为数据准备的相关代码：

df=pd.read_csv('iris_full.csv')  
data = np.array(df)  
X = data[0:,1:5]  
Y = data[0:,-1]  
np.random.shuffle(data)  
num_training_samples = 120

# 其中x_train,y_train为训练集的特征以及类别，x_test,y_test为测试集的特征以及类别  
x_train = data[0:num_training_samples,1:5].astype(float)  
y_train = data[0:num_training_samples,-1]  
x_test = data[num_training_samples:150,1:5].astype(float)  
y_test = data[num_training_samples:150,-1]  
encoder = LabelEncoder()  
Y_encoded = encoder.fit_transform(y_train)

y_train = np_utils.to_categorical(Y_encoded)

# 二、构建模型

本实验中使用的是一个基本的前馈神经网络模型，模型包含了输入层、隐藏层和输出层。选定神经元数目，层数、激活函数、损失函数、优化算法等设定建立模型。

## 2.1模型1的建立

输入层：包含4个节点，代表每个莺尾花的4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），因此input_dim=4,采用常用的relu激活函数：

model.add(Dense(10,activation='relu',input_dim=4))

输出层：包含3个节点，代表3个莺尾花类别。使用课上讲过的Softmax激活函数将输出转换为概率：

model.add(Dense(3, activation='softmax'))

## 2.2 模型2的建立

在模型1的基础上添加一个神经网络层，提高模型的复杂度，采用10个神经元，激活函数仍然采用relu函数：

model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.05), input_dim=4))

## 2.3 模型3的建立

在模型2的基础上添加一个包括8个神经元，使用relu函数的神经网络层：

model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.05)))

## 2.4 模型4的建立

当前模型已经在模型1的基础上添加了神经元个数为10、8的神经网络层，因此本次模型在模型3的基础上添加一个包括6个神经元，使用relu函数的神经网络层：

model.add(Dense(6, activation='relu', kernel_regularizer=regularizers.l2(0.05)))

## 2.5 模型5的建立

尝试了不同数目的神经元来构建神经网络层之后，在模型4的基础上将激活函数由relu函数变为tanh函数，tanh函数在输入值较大或较小的情况下，tanh函数具有更陡峭的斜率，这使得它能够更快地学习并更好地解决梯度消失问题。模型5的代码如下：

model = Sequential()

model.add(Dense(10,activation='tanh',input_dim=4))

model.add(Dense(10, activation='tanh', kernel_regularizer=regularizers.l2(0.05), input_dim=4))

model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.05)))

model.add(Dense(6, activation='tanh', kernel_regularizer=regularizers.l2(0.05)))

model.add(Dense(3, activation='softmax'))

# 三、模型的编译以及训练

本次实验采用keras中的compile函数进行模型的编译，采用了机器学习中常用的Adam优化器，Adam算法在梯度下降法的基础上，通过动态调整学习率来提高训练效果。损失函数采用了分类交叉熵损失函数将训练集送入模型进行训练，衡量模型输出的概率分布与实际标签的差异，并对错误的分类给出较大的惩罚，方便优化模型。模型编译代码如下：

model.compile(optimizer=adam.Adam(learning_rate=0.008),loss='categorical_crossentropy', metrics=['accuracy'])

关于模型的训练，本次实验采用了keras自带的fit函数进行训练，使用了50个epochs，每个batch大小为20：

history = model.fit(x_train, y_train,  epochs=50, batch_size=20,verbose =1)

# 四、监控训练过程

在训练过程中，调用fit()方法返回的log_training对象中有一个成员history，记录了训练集和测试集上的损失函数和准确率，通过导入matplotlib中的pyplot，绘制了训练集和测试集上的准确率和损失函数的变化曲线：

plt.plot(history.history['loss'], label='Training Loss')

plt.xlabel('Epoch')

plt.legend()

plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.xlabel('Epoch')

plt.legend()

plt.show()

# 五、评估模型

在测试集上进行测试，计算模型在测试集上的准确率。通过调整模型的不同参数，如隐藏层数量、神经元个数、优化算法等，比较不同模型在测试集上的准确率。

模型1的准确率和损失函数的变化曲线以及在测试集上的准确率如下：

![图表
低可信度描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image002.png) ![图表, 折线图
描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image004.png)

test_accuracy =0.9333333333333333

模型2的准确率和损失函数的变化曲线以及在测试集上的准确率如下：

![图片包含 图形用户界面
描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image006.png) ![图片包含 折线图
描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image008.png)

test_accuracy = 0.9666666666666667

模型3的准确率和损失函数的变化曲线以及在测试集上的准确率如下：

![图片包含 形状
描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image010.png) ![图形用户界面
低可信度描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image012.png)

test_accuracy =0.9333333333333333

模型4的准确率和损失函数的变化曲线以及在测试集上的准确率如下：

![](file:///D:/TEMP/msohtmlclip1/01/clip_image014.png) ![图片包含 图形用户界面
描述已自动生成](file:///D:/TEMP/msohtmlclip1/01/clip_image016.png)

test_accuracy =0.9333333333333333

模型5的准确率和损失函数的变化曲线以及在测试集上的准确率如下：

![](file:///D:/TEMP/msohtmlclip1/01/clip_image018.png) ![](file:///D:/TEMP/msohtmlclip1/01/clip_image020.png)

test_accuracy =0.9666666666666667

以上是训练结果的可视化，可以看出训练集和测试集上的损失函数和准确率随着迭代次数的增加而逐渐收敛，并且测试集的表现和训练集的表现非常接近，说明模型没有过拟合。

可以看出，从模型1到模型4，随着加入神经网络层数的增多，识别准确率达到稳定的过程有所加快，稳定后的波动逐渐稳定。这也表明神经网络层数的增多，有利于提高模型训练的速度与模型的稳定性。

而模型4和模型5采用了不同的激活函数，可以看出，无论是识别的准确率、模型训练的速度还是模型的稳定性，模型5采用的tanh函数在处理莺尾花的识别问题上更有优势。