
from subfunction import *
class generator:
    def __init__(self,name):
        self.name=name

    def __call__(self, inputs,nums=6,G=32,G_0=64):
        with tf.variable_scope(self.name):
            inputs1_1 = leaky_relu(conv("conv1.1", inputs, 64, 3))
            inputs1_2 = leaky_relu(conv("conv1.2", inputs, 64, 5))
            inputs1_3 = leaky_relu(conv("conv1.3", inputs, 64, 7))
            inputs1_4 = leaky_relu(conv("conv1.4", inputs, 64, 9))
            inputs1_5 = leaky_relu(conv("conv1.5", inputs, 64, 11))
            inputs_concate=conv('concate',leaky_relu(tf.concat([inputs1_1,inputs1_2,inputs1_3,inputs1_4,inputs1_5 ],axis=-1)),64,1)
            inputs1 = RDB_("R2", inputs1_1,C_nums=nums, input_num=G, out_num=G_0)
            inputs2 = leaky_relu(tf.concat([inputs1, inputs1_2], -1))
            inputs3 = RDB("R3", inputs2,inputs1_2, C_nums=nums, input_num=G, out_num=G_0)
            inputs3 = leaky_relu(tf.concat([inputs3, inputs1_3], -1))
            inputs4 = RDB("R4", inputs3,inputs1_3, C_nums=nums, input_num=G, out_num=G_0)
            inputs4 = leaky_relu(tf.concat([inputs4, inputs1_4], -1))
            inputs5 = RDB("R5", inputs4,inputs1_4,C_nums= nums, input_num=G, out_num=G_0)
            inputs5 = leaky_relu(tf.concat([inputs5, inputs1_5], -1))
            inputs6 = RDB("R6", inputs5, inputs1_5,C_nums=nums, input_num=G, out_num=G_0)
            inputs =inputs_concate + inputs6
            inputs = pixelShuffler(inputs, 2)
            inputs = relu(conv("Up_conv1", inputs, G, 3, 1))
            inputs = pixelShuffler(inputs, 2)
            inputs = conv("Up_conv2", inputs, 3, 3, 1)
        return tf.nn.tanh(inputs)
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
class discriminator:
    def __init__(self, name):
        self.name = name
    def __call__(self, inputs, down_inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs=DownBlock("Down1",inputs,3,64)
            inputs=DownBlock("Down1_1",inputs,3,64,False)
            inputs=DownBlock("Down2",inputs,3,128)
            inputs=DownBlock("Down2_1",inputs,3,128,False)
            x=Inner_product(inputs,down_inputs)
            inputs=DownBlock("Down3",inputs,3,128)
            inputs=DownBlock("Down4",inputs,3,256)
            inputs=DownBlock("Down5",inputs,3,512)
            inputs=DownBlock("Down6",inputs,3,1024)
            inputs=DownBlock('Down7',inputs,3,1024,False)
            inputs = leaky_relu(inputs)
            inputs=global_sum_pooling(inputs)
            inputs=tf.add(Linear("Linear",inputs,1024,1),x)
        return inputs
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
