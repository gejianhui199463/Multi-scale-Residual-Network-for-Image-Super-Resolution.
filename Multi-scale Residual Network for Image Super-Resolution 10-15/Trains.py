import time
import argparse
from Tests import *

def train(batch_size=32,lambd=100, init_lr=2e-4,  max_itr=1000000,
          path_trainset="", path_vgg="./vgg_para/",
          path_save_model="./save_para/"):
    inputs = tf.placeholder(tf.float32, [None,128, 128, 3])
    downsampled = tf.placeholder(tf.float32, [None, 32, 32, 3])
    learning_rate=tf.placeholder(tf.float32,)
    G = generator("generator")
    D = discriminator("discriminator")
    SR = G(downsampled)
    fake_logits = D(SR, downsampled)
    real_logits = D(inputs, downsampled)
    D_loss = tf.reduce_mean(tf.maximum(0., 1 - real_logits)) + tf.reduce_mean(tf.maximum(0., 1 + fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    MSE=tf.reduce_mean(tf.reduce_sum(tf.abs(SR - inputs), axis=[1, 2, 3]))
    G_loss += MSE * lambd
    itr = tf.Variable(max_itr, dtype=tf.int32, trainable=False)
    op_sub = tf.assign_sub(itr, 1)
    D_opt = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())
    with tf.control_dependencies([op_sub]):
        G_opt = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9).minimize(G_loss, var_list=G.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\para.ckpt")
    lr0 = init_lr
    while True:
        s0 = time.time()
        batch, down_batch = read_crop_data(path_trainset, batch_size, [128, 128, 3], 4)
        e0 = time.time()
        s1 = time.time()
        sess.run(D_opt, feed_dict={inputs: batch, downsampled: down_batch,  learning_rate: lr0})
        [_,iteration]=sess.run([G_opt,itr],feed_dict={inputs: batch, downsampled: down_batch, learning_rate: lr0})
        iteration_ = iteration * 1.0
        iteration = max_itr - iteration
        e1 = time.time()
        if iteration > max_itr // 2:
            learning_rate=lr0 *(iteration_*2/max_itr)
        if iteration % 100 == 0:
            [d_loss, g_loss, sr] = sess.run([D_loss, G_loss, SR], feed_dict={downsampled: down_batch, inputs: batch})
            raw = np.uint8((batch[-1,:,:,:] + 1) * 127.5)
            bicub = misc.imresize(np.uint8((down_batch[-1,:,:,:] + 1) * 127.5), [128, 128])
            gen = np.uint8((sr[-1, :, :, :] + 1) * 127.5)
            print("Iteration: %d, D_loss: %f, G_loss: %e, PSNR: %f, SSIM: %f, Read_time: %f, Update_time: %f" %
                      (iteration, d_loss, g_loss, psnr(raw, gen),
                    ssim(raw, gen, multichannel=True), e0 - s0, e1 - s1))
            Image.fromarray(np.concatenate((raw, bicub, gen), axis=1)).save("./Results/" + str(iteration) + ".jpg")
        if iteration % 500 == 0:
            saver.save(sess, path_save_model+"para.ckpt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=1e-14)
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--clip_v", type=float, default=0.05)
    parser.add_argument("--max_itr", type=int, default=1000000)
    parser.add_argument("--path_trainset", type=str, default="")
    parser.add_argument("--path_vgg", type=str, default="./vgg_para/")
    parser.add_argument("--path_save_model", type=str, default="./save_para/")
    parser.add_argument("--is_trained", type=bool, default=False)
    args = parser.parse_args()
    if args.is_trained:
        parser.add_argument("--path_test_img", type=str, default="//")
        args = parser.parse_args()
        img = np.array(Image.open(args.path_test_img))
        h, w = img.shape[0] // 4, img.shape[1] // 4
        downsampled_img = misc.imresize(img, [h, w])
        test(downsampled_img, img,)
    else:
        train(batch_size=args.batch_size,lambd=args.lambd, init_lr=args.learning_rate,
              max_itr=args.max_itr, path_trainset=args.path_trainset, path_vgg=args.path_vgg,
              path_save_model=args.path_save_model)



