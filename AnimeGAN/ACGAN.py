import os 
import sys
import numpy as np
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True) 
                             ,device_count={'GPU': 1})


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import io

logdir="AC_WGANGP-ver0"

if not( os.path.exists("./samples") ):
    os.mkdir("./samples")



processing_input = lambda x : (x/255-0.5)*2
inverse_processing = lambda x : (x/2+0.5)*255

model_path = "./model_para/{}/generator_{}.ckpt.meta"
saver = tf.train.import_meta_graph(model_path.format(logdir,100000))

sess = tf.Session(config=gpu_opt)

g = sess.graph

for op in g.get_operations():
    if "train_strategy" in op.name or "Gradient" in op.name:
        continue
    if "Input/latent_space" in op.name:
        z_input = op.outputs[0]
    elif 'Generator/g_4/Tanh' in op.name:
        g_img = op.outputs[0]
    ## hair has 12 tags , eyes have 10 tags
    ## placeholder 12 , placeholder_1 10
    elif 'Class/Placeholder' in op.name:
        if "_1" in op.name:
            eyes_tag = op.outputs[0]
        else:
            hair_tag = op.outputs[0]
            
            
saver.restore(sess , model_path.format(logdir,75000)[0:-5])

cor_z = np.load("tmp/AC_WGANGP-ver0/z_vector.npy")[0]
z = cor_z["z"]
z_hair = cor_z["hair"]
z_eyes = cor_z["eyes"]
#################---------------reproduce---------------#################
tmp_image = inverse_processing(sess.run(g_img , feed_dict={z_input:z , hair_tag:z_hair , eyes_tag:z_eyes}))


r = 5
c = 5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
#             tmp = img_resize( gen_imgs[cnt, :,:,:], [64,64] )
        tmp = tmp_image[cnt]
        axs[i,j].imshow(tmp.astype("uint8"))
        axs[i,j].axis('off')
        cnt += 1

fig.savefig("./samples/cgan.png")

#################---------------End---------------#################



z = np.random.randn(25,int(z_input.shape[-1]))
np.save("tmp/test_z_vector.npy" , z)

tags_path = sys.argv[1]
tags_file = np.genfromtxt(tags_path , dtype="str" , delimiter=",")

code2eyes = ['aqua', 'blac', 'blue', 'brow', 'gree', 'oran', 'pink', 'purp','red', 'yell']
code2hair = ['aqua', 'blac', 'blon', 'blue', 'brow', 'gray', 'gree', 'oran','pink', 'purp', 'red', 'whit']

hair2code = dict([(s , i) for i , s in enumerate(code2hair) ])
eyes2code = dict([(s , i) for i , s in enumerate(code2eyes) ])


def split_tag(x):
    x = x[1].strip().split(" ")
    x = [x[0][0:4] , x[2][0:4] ]
    return x

image_tag = np.apply_along_axis(split_tag , arr=tags_file , axis=1)
hair_style = image_tag[:,0]
eyes_style = image_tag[:,1]

z_hair , z_eyes = [] , []
for h , e in zip(hair_style , eyes_style):
    print( h , e )
    z_hair.append(hair2code[h])
    z_eyes.append(eyes2code[e])


tmp_image = inverse_processing(sess.run(g_img , feed_dict={z_input:z , hair_tag:z_hair , eyes_tag:z_eyes}))


r = 5
c = 5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
#             tmp = img_resize( gen_imgs[cnt, :,:,:], [64,64] )
        tmp = tmp_image[cnt]
        axs[i,j].imshow(tmp.astype("uint8"))
        axs[i,j].axis('off')
        cnt += 1

fig.savefig("./samples/cgan.png")





















