import os 
import numpy as np
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True) 
                             ,device_count={'GPU': 1})


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import io

logdir="WGANGP-ver0"

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
            
            
saver.restore(sess , model_path.format(logdir,85000)[0:-5])

z = np.load("tmp/{}/output_WGAN_128.npy".format(logdir))


tmp_image = inverse_processing(sess.run(g_img , feed_dict={z_input:z}))


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

fig.savefig("./samples/gan_original.png")





















