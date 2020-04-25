from networkdesen import *
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

def up_scale(downsampled_img):
    downsampled_img = downsampled_img[np.newaxis, :, :, :]
    downsampled = tf.placeholder(tf.float32, [None, None, None, 3])
    G = generator("generator")
    SR = G(downsampled)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\para.ckpt")
    SR_img = sess.run(SR, feed_dict={downsampled: downsampled_img/127.5 - 1})
    Image.fromarray(np.uint8((SR_img[0, :, :, :] + 1)*127.5))
    Image.fromarray(np.uint8((downsampled_img[0, :, :, :])))
    sess.close()
def test(downsampled_img, img):
    downsampled_img = downsampled_img[np.newaxis, :, :, :]
    downsampled = tf.placeholder(tf.float32, [None, None, None, 3])
    G = generator("generator")
    SR = G(downsampled)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\para.ckpt")
    SR_img = sess.run(SR, feed_dict={downsampled: downsampled_img/127.5 - 1})
    Image.fromarray(np.uint8((SR_img[0, :, :, :] + 1) * 127.5)).show()
    Image.fromarray(np.uint8((SR_img[0, :, :, :] + 1)*127.5)).save("C://Users//asus//Desktop//imageFigure8-12//srGAN97531.jpg")
    Image.fromarray(np.uint8((downsampled_img[0, :, :, :]))).show()
    h = img.shape[0]
    w = img.shape[1]
    bic_img = misc.imresize(downsampled_img[0, :, :, :], [h, w])
    Image.fromarray(np.uint8((bic_img))).show()
    SR_img = misc.imresize(SR_img[0, :, :, :], [h, w])
    p = psnr(img, SR_img)
    s = ssim(img, SR_img, multichannel=True)
    p1 = psnr(img, bic_img)
    s1 = ssim(img, bic_img, multichannel=True)
    print("SR PSNR: %f, SR SSIM:%f, BIC PSNR: %f, BIC SSIM: %f"%(p, s, p1, s1))
    sess.close()

