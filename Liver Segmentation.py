# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
from matplotlib.image import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
data_path = r"C:\Users\user\OneDrive\Desktop\atlas-train-dataset-1.0.1-20240117T200737Z-001\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train\imagesTr"
train_data = []
target_size = (256, 256)
for elem in range(len(os.listdir(data_path))):
  s = sorted(os.listdir(data_path))[elem]
  l = nib.load(data_path+'/'+s)
  f = l.get_fdata()
  for i in range(f.shape[2]):
    v = str(i)
    slice_data = f[:, :, i]
    resized_slice = resize(slice_data, target_size, order=1, anti_aliasing=True)
    train_data.append(resized_slice)

label_path = r"C:\Users\user\OneDrive\Desktop\atlas-train-dataset-1.0.1-20240117T200737Z-001\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train\labelsTr"
train_label = []
for elem in range(len(os.listdir(label_path))):
  s = sorted(os.listdir(label_path))[elem]
  l = nib.load(label_path+'/'+s)
  f = l.get_fdata()
  for i in range(f.shape[2]):
    v = str(i)
    slice_data = f[:, :, i]
    resized_slice = resize(slice_data, target_size, order=1, anti_aliasing=True)
    train_label.append(resized_slice)

print(len(train_data))
print(len(train_label))

train_data = np.array(train_data)
train_label = np.array(train_label)


print(train_data.shape)
print(train_label.shape)
train_data= train_data.reshape(-1,256,256,1)
train_label= train_label.reshape(-1,256,256,1)
print(train_data.shape)
print(train_label.shape)


train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
# Split the indices into train, validation, and test sets
train_indices, test_val_indices = train_test_split(np.arange(train_data.shape[0]), test_size=(val_ratio + test_ratio), random_state=42)
val_indices, test_indices = train_test_split(test_val_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

# Split data and labels based on the indices
train_images = train_data[train_indices]
train_labels = train_label[train_indices]

val_images = train_data[val_indices]
val_labels = train_label[val_indices]

test_images = train_data[test_indices]
test_labels = train_label[test_indices]

print(train_images.shape)
print(val_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)


'''def dice_coef_metric(pred, target):
    smooth = 1e-5
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(target, [-1])
    intersection = tf.reduce_sum(pred_flat * target_flat)
    A_sum = tf.reduce_sum(pred_flat * pred_flat)
    B_sum = tf.reduce_sum(target_flat * target_flat)
    dice_score = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)

    return dice_score'''

'''def dice_loss(pred, target):
    smooth = 1e-5
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(target, [-1])
    intersection = tf.reduce_sum(pred_flat * target_flat)
    A_sum = tf.reduce_sum(pred_flat * pred_flat)
    B_sum = tf.reduce_sum(target_flat * target_flat)
    dice_loss_value = 1.0 - (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)

    return dice_loss_value

def dice_loss_all(pred, target, weight=None):
    if weight is None:
        weight = [1.0, 1.0, 1.0]

    loss_backgr = dice_loss(pred[:, :, 0], target[:, :, 0])
    loss_myo = dice_loss(pred[:, :, 1], target[:, :, 1])
    loss_endocard = dice_loss(pred[:, :, 2], target[:, :, 2])

    # Average loss
    avg_loss = (weight[1] * loss_myo + weight[2] * loss_endocard) / 2.0

    return avg_loss'''


def batch_Norm_Activation(x, BN=False): ## To Turn off Batch Normalization, Change BN to False >
    if BN == True:
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    else:
        x= Activation("relu")(x)
    return x

input_layer = Input(shape=(256, 256, 1)) ##x_train have shape of (no of images, H, W, Channels), after applying this --> (none, height, width, channels)
print(input_layer.shape)
print(input_layer.dtype)



def conv_block(x_in, filters, batch_norm, kernel_size=(3,3)): 
    x = Conv2D(filters, kernel_size, padding='same')(x_in)
    if batch_norm=='TRUE':
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    if batch_norm=='TRUE':
        x = BatchNormalization()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_2d(x_in, filters, batch_norm, kernel_size=(3,3),acti ='relu'):
    x = Conv2D(filters, kernel_size, padding='same')(x_in)
    if batch_norm=='TRUE':
        x=BatchNormalization()(x)
    x= Activation(acti)(x)
    return x
    
def pool(x_in, pool_size=(2, 2), type='Max'):
    if type=='Max':
        p = MaxPooling2D(pool_size)(x_in)
    return p

def up(x_in, filters, merge, batch_norm, size=(2,2)):
    u = UpSampling2D(size)(x_in)
    conv = conv_block(u, filters, batch_norm)
    merge=concatenate([merge, conv],axis=-1)
    return merge
    
def Unet_05(input_layer):
    conv1 = conv_block(input_layer, filters=16, batch_norm='TRUE')
    pool1 = pool(conv1)
    
    conv2 = conv_block(pool1, filters=32, batch_norm='TRUE')
    pool2 = pool(conv2)
    
    conv3 = conv_block(pool2, filters=32, batch_norm='TRUE')
    pool3 = pool(conv3)
    
    conv4 = conv_block(pool3, filters=64, batch_norm='TRUE')
    pool4 = pool(conv4)
    
    conv5 = conv_2d(pool4, filters=128, batch_norm='TRUE')
    
    up1 = up(conv5,filters=128, merge=conv4, batch_norm='TRUE')
    conv6 = conv_2d(up1, filters=128, batch_norm='TRUE')
    
    up2 = up(conv6, filters=128, merge=conv3, batch_norm='TRUE')
    conv7 = conv_2d(up2, filters=128, batch_norm='TRUE')
    
    up3 = up(conv7, filters=64, merge=conv2, batch_norm='TRUE')
    conv8 = conv_2d(up3, filters=64, batch_norm='TRUE')
    
    up4 = up(conv8, filters=32, merge=conv1, batch_norm='TRUE')
    conv9 = conv_2d(up4, filters=32, batch_norm='TRUE')
    
    conv10 = conv_2d(conv9, filters=1, batch_norm='FALSE', acti='sigmoid')
    
    output_layer = conv10
    model = Model(input_layer, output_layer)
    
    return model

model = Unet_05(input_layer)
model.compile(optimizer=Adam(), loss= 'categorical_crossentropy', metrics=['Accuracy'])
model.summary()

model.fit(train_images, train_labels, batch_size=4, epochs=50, validation_data = (val_images, val_labels), verbose=1)
