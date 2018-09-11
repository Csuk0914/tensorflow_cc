import os
 
import tensorflow as tf
 
from six.moves import xrange
 
#原图像的尺度为32*32,但根据常识，信息部分通常位于图像的中央，这里定义了中心裁剪后图像的尺寸
image_size=24
 
num_classes=10
 
num_examples_per_epoch_for_train=50000#测试集实例
 
num_examples_per_epoch_for_eval=10000#训练集实例
 
def read_cifar10(filename_queue):
    class cifar10recode(object):
        pass
    result=cifar10recode()
 
    label_bytes = 1
 
    result.height=32
    result.width=32
    result.depth=3
 
    image_bytes=result.height * result.width * result.depth
 
    recode_bytes=label_bytes+image_bytes
 
    reader=tf.FixedLengthReader(recode_bytes=recode_bytes)
    #每次从文件中读取固定字节数
 
    result.key,value=reader.read(filename_queue)#????键值是什么意思？？？
    #返回从filename_queue中读取的(key,value)对，key和value都是字符串类型的Tensor
 
    recode_bytes=tf.decode_row(value,tf.unit8)
    #解码操作可以看做读二进制文件，把字符串中的字节转换为数值向量，每一个数值占用一个字节，在[0,255]区间内，因此要去unit8类型
 
    result.label=tf.cast(tf.slice(recode_bytes,[0],[label_bytes]),tf.int32)
    #begin=[0],size=[label_bytes]  begin和size分别表示待截取片段的起点和长度 
    #tf.cast():将数组转化为指定的数据类型 tf.slice():对tensor进行切片操作
    #从一维tensor对象中截取一个slice,类似于从一维向量中筛选子向量
    #因为value中包含了label和feature，故要对向量类型tensor进行'parse'操作（解析操作）
 
    depth_major=tf.reshape(tf.slice(recode_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    #这里的维度顺序是依靠cifar二进制文件的格式而定的
    result.unit8image=tf.transpose(depth_major,[1,2,0])
    #对depth_major的维度重新排列 [depth, height, width] 转换成 [height, width, depth].
    #为什么要转化？？？为什么不在reshape处就这样排列？？？和训练集自身关系？？？
    
    
    
def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size):
    num_preprocess_threads=16
    images,label_batch=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=num_preprocess_threads,
                                              capacity=min_queue_example+3*batch_size,
                                              min_after_dequeue=min_queue_examples)
    '''
    tf.train.shuffle_batch()函数用于随机地shuffling 队列中的tensors来创建batches(也即每次可以读取多个data文件中的样例
    构成一个batch)。这个函数向当前Graph中添加了下列对象： 
    *创建了一个shuffling queue，用于把‘tensors’中的tensors压入该队列； 
    *一个dequeue_many操作，用于根据队列中的数据创建一个batch； 
    *创建了一个QueueRunner对象，用于启动一个进程压数据到队列 
    capacity参数用于控制shuffling queue的最大长度；
    min_after_dequeue参数表示进行一次dequeue操作后队列中元素的最小数量，
    可以用于确保batch中元素的随机性；
    num_threads参数用于指定多少个threads负责压tensors到队列；
    enqueue_many参数用于表征是否tensors中的每一个tensor都代表一个样例 
    tf.train.batch()与之类似，只不过顺序地出队列（也即每次只能从一个data文件中读取batch），少了随机性。'''
    tf.image_summary('images',images)
    #输出预处理后图像的summary缓存对象，用于在session中写入事件文件中，tensorboard中用得到
    return images,tf.reshape(label_batch,[batch_size])
 
def distorted_input(data_dir,batch_size):
    '''这部分程序用于对训练数据集进行‘数据增强’操作，通过增加训练集的大小来防止过拟合''' 
    filenames=[os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
    #os.path.join(path, name)——连接目录和文件名 数据集一共5个文件
    for f in filenames:
        if not tf.grile.Exists(f): #检验训练数据集文件是否存在  
            raise ValueError('Failed to find file :'+ f)
 
        filename_queue=tf.train.string_input+producer(filenames)
        # 把文件名输出到队列中，作为整个data pipe的第一阶段 
        read_input=read_cifar10(filename_queue)
        reshaped_image=tf.cast(read_input.unit8image,tf.float32)
 
        height=image_size
        width=image_size
 
        distorted_image=tf.random_crop(reshape_image,[height,width,3])#从原图像中切割出子图像  
        distorted_image=tf.image.random_flip_left_right(distorted_image) #随机地左右翻转图像
        distorted_images=tf.image.random_brightness(distorted_image,max_delta=63)#随机调节图像的亮度
        distorted_image=tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)#随机地调整图像对比度
        float_image=tf.image.per_image_whitening(distorted_image)
      #对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输入特征间的相关性
        min_fraction_of_examples_in_queue=0.4 
       #用于确保读取到的batch中样例的随机性，使其覆盖到更多的类别、更多的数据文件！！！
        min_queue_examples=int(num_examples_per_epoch_for_train*min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR image before starting to train,this will take a few minutes.'% min_queue_examples)
        return _generate_image_and_label_batch(foat_image,read_input.label,min_queue_examples,batch_size)
 
def input(eval_data,data_dir,batch_size):
    #和前一个函数的作用类似
    filenames=[os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
    #os.path.join(path, name)——连接目录和文件名 数据集一共5个文件
    for f in filenames:
        if not tf.grile.Exists(f): #检验训练数据集文件是否存在  
            raise ValueError('Failed to find file :'+ f)
 
        filename_queue=tf.train.string_input+producer(filenames)
        # 把文件名输出到队列中，作为整个data pipe的第一阶段 
        read_input=read_cifar10(filename_queue)
        reshaped_image=tf.cast(read_input.unit8image,tf.float32)
 
        height=image_size
        width=image_size
 
        resized_image=tf.image.resize_image_with_crop_or_pad(reshape_images,width,height)# 截取图片中心区域
        float_image=tf.image.per_image_whitening(resized_image)
        min_fraction_of_examples_in_queue=0.4 #用于确保读取到的batch中样例的随机性，使其覆盖到更多的类别、更多的数据文件！！！
        min_queue_examples=int(num_examples_per_epoch_for_train*min_fraction_of_examples_in_queue)
        return _generate_image_and_label_batch(foat_image,read_input.label,min_queue_examples,batch_size)
