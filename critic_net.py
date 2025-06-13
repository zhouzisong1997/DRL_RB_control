#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf_2


class critic:

    def __init__(self, session, input_size_critic_s, input_size_critic_a, output_size_critic, actor, name="main"): 

        self.session = session

        self.input_size_critic_s = input_size_critic_s
        self.input_size_critic_a = input_size_critic_a
        self.output_size_critic = output_size_critic
 
        self.input_critic_state = tf.placeholder(tf.float32, [None, self.input_size_critic_s[0]*self.input_size_critic_s[1]*self.input_size_critic_s[2]], \
                                                name="input_critic_state")    
        self.feed_action = tf.placeholder(tf.float32, [None, self.input_size_critic_a[0]*self.input_size_critic_a[1]*self.input_size_critic_a[2]], name="feed_action")
        self.action_pred = actor.action_pred
        print("action_pred")
        print(self.action_pred)
        
        self.total_channel = self.input_size_critic_s[2] + self.input_size_critic_a[2]
        
        self.net_name = name


        print(name)
        self.Q_pred = self.build_network (self.feed_action, "Critic_net connected - for action_feed") 
        print(name)
        self.Objective = self.build_network (self.action_pred, "Critic_net connected - for actor_feed") 
        
    
    
    def build_network(self, action, sentence, h_size = 32, k_1size = 32, k_2size=32, k_3size = 32, k_4size = 32, k_5size=32, k_6size = 32, pooling = 2): #, h_size=10, l_rate=0.001

        
        with tf.variable_scope(self.net_name, reuse=tf.AUTO_REUSE): 

            self.input_critic_action = action 
            self.X_state_input = self.input_critic_state 

            X_input_1 = tf.reshape(self.X_state_input, [-1, self.input_size_critic_s[0], self.input_size_critic_s[1], self.input_size_critic_s[2]])      
            X_input_2 = tf.reshape(self.input_critic_action, [-1, self.input_size_critic_a[0], self.input_size_critic_a[1], self.input_size_critic_a[2]])    
            X_input= tf.concat([ X_input_1, X_input_2], -1)
            self.X_state_input_2 = tf.reshape(X_input, [-1, self.input_size_critic_s[0], self.input_size_critic_s[1], self.total_channel]) 
            

            # Convolution layer 1
            k1_S = [3, 3, self.total_channel, k_1size]
            k1_std = np.sqrt(2)/np.sqrt(np.prod(k1_S[:-1]))
            
            self.kernal_1 = tf.get_variable("W_c1", shape =[3, 3, self.total_channel, k_1size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k1_std, seed=None))
            self.B_c1 = tf.get_variable("B_c1", shape = [1,1, 1, k_1size], initializer = tf.initializers.zeros())

            linear_1 = tf.nn.conv2d(self.X_state_input_2, self.kernal_1, strides=[1, 1, 1, 1], padding='SAME') + self.B_c1
            active_1 = tf.nn.relu(linear_1)

            
            # Convolution layer 2
            k2_S = [3, 3, k_1size, k_2size]
            k2_std = np.sqrt(2)/np.sqrt(np.prod(k2_S[:-1]))
            
            self.kernal_2 = tf.get_variable("W_c2", shape =[3, 3, k_1size, k_2size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k2_std, seed=None))
            self.B_c2 = tf.get_variable("B_c2", shape = [1,1, 1, k_2size], initializer = tf.initializers.zeros())
            linear_2 = tf.nn.conv2d(active_1, self.kernal_2, strides=[1, 1, 1, 1], padding='SAME') + self.B_c2
            active_2 = tf.nn.relu(linear_2)
            conv_2 = tf.nn.avg_pool(active_2, ksize=[1, 2, 2, 1], strides=[1, pooling, pooling, 1], padding='SAME')


            # Convolution layer 3
            k3_S = [3, 3, k_2size, k_3size]
            k3_std = np.sqrt(2)/np.sqrt(np.prod(k3_S[:-1]))
            
            self.kernal_3 = tf.get_variable("W_c3", shape =[3, 3, k_2size, k_3size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k3_std, seed=None))
            self.B_c3 = tf.get_variable("B_c3", shape = [1,1, 1, k_3size], initializer = tf.initializers.zeros())
            linear_3 = tf.nn.conv2d(conv_2, self.kernal_3, strides=[1, 1, 1, 1], padding='SAME') + self.B_c3
            active_3 = tf.nn.relu(linear_3)


            # Convolution layer 4
            k4_S = [3, 3, k_3size, k_4size]
            k4_std = np.sqrt(2)/np.sqrt(np.prod(k4_S[:-1]))
            
            self.kernal_4 = tf.get_variable("W_c4", shape =[3, 3, k_3size, k_4size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k4_std, seed=None))
            self.B_c4 = tf.get_variable("B_c4", shape = [1,1, 1, k_4size], initializer = tf.initializers.zeros())
            linear_4 = tf.nn.conv2d(active_3, self.kernal_4, strides=[1, 1, 1, 1], padding='SAME') + self.B_c4
            active_4 = tf.nn.relu(linear_4)
            conv_4 = tf.nn.avg_pool(active_4, ksize=[1, 2, 2, 1], strides=[1, pooling, pooling, 1], padding='SAME')

            # Convolution layer 5
            k5_S = [3, 3, k_4size, k_5size]
            k5_std = np.sqrt(2)/np.sqrt(np.prod(k5_S[:-1]))
            
            self.kernal_5 = tf.get_variable("W_c5", shape =[3, 3, k_4size, k_5size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k5_std, seed=None))
            self.B_c5 = tf.get_variable("B_c5", shape = [1,1, 1, k_5size], initializer = tf.initializers.zeros())
            linear_5 = tf.nn.conv2d(conv_4, self.kernal_5, strides=[1, 1, 1, 1], padding='SAME') + self.B_c5
            active_5 = tf.nn.relu(linear_5)

            # Convolution layer 6
            k6_S = [3, 3, k_6size, k_6size]
            k6_std = np.sqrt(2)/np.sqrt(np.prod(k6_S[:-1]))
            
            self.kernal_6 = tf.get_variable("W_c6", shape =[3, 3, k_5size, k_6size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k6_std, seed=None))
            self.B_c6 = tf.get_variable("B_c6", shape = [1,1, 1, k_6size], initializer = tf.initializers.zeros())
            linear_6 = tf.nn.conv2d(active_5, self.kernal_6, strides=[1, 1, 1, 1], padding='SAME') + self.B_c6
            active_6 = tf.nn.relu(linear_6)
            conv_6 = tf.nn.avg_pool(active_6, ksize=[1, 2, 2, 1], strides=[1, pooling, pooling, 1], padding='SAME') #thrid pooling

                         
            conv_7_flat = tf.reshape(conv_6, [-1, int(self.input_size_critic_s[0]/pooling**3)*int(self.input_size_critic_s[1]/pooling**3)*k_3size]) 
            
            
            #Fully connected layers
            W7_s = [int(self.input_size_critic_s[0]/pooling**3)*int(self.input_size_critic_s[1]/pooling**3)*k_6size , h_size]
            W7_std = np.sqrt(2)/np.sqrt(np.prod(W7_s[:-1]))
            self.W_c7 = tf.get_variable("W_c7", shape = W7_s, initializer = tf.random_normal_initializer(mean=0.0, stddev=W7_std, seed=None))
            self.B_c7 = tf.get_variable("B_c7", shape = [1,h_size], initializer = tf.initializers.zeros())

            layer7 = tf.matmul(conv_7_flat, self.W_c7) + self.B_c7
            active7 = tf.nn.relu(layer7)
            
            #Fully connected layers
            W8_s = [h_size, self.output_size_critic]
            W8_std = np.sqrt(1)/np.sqrt(np.prod(W8_s[:-1]))

            self.W_c8 = tf.get_variable("W_c8", shape = W8_s, initializer = tf.random_normal_initializer(mean=0.0,stddev=W8_std, seed=None))   
            self.B_c8 = tf.get_variable("B_c8", shape = [1, self.output_size_critic], initializer = tf.initializers.zeros())
            layer8 = tf.matmul(active7, self.W_c8) + self.B_c8
        
            Q_pred = 0.1*layer8
              
            self.Q_target = tf.placeholder(shape=[None, self.output_size_critic], dtype = tf.float32)          
             
            print(sentence)
        
        return Q_pred
        
    def initialization_c (self, name ="ops_name", l_rate=0.002, B = 0.0001):       
            
            self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            
            self.regular = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_vars if 'W' in v.name])
   
            self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_pred)) + B*self.regular

            self.train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.loss, var_list = self.critic_vars)
            
            
            
    def predict(self, state, action): 
        input_state = np.reshape(state, [1, self.input_size_critic_s[0]*self.input_size_critic_s[1]*self.input_size_critic_s[2]])
        input_action = np.reshape(action, [1, self.input_size_critic_a[0]*self.input_size_critic_a[1]*self.input_size_critic_a[2]])

        return self.session.run(self.Q_pred, feed_dict={self.input_critic_state: input_state, self.feed_action: input_action})
                 
    def update(self, x_state_stack, x_action_stack, y_stack):
        feed = {self.input_critic_state: x_state_stack, self.feed_action: x_action_stack, self.Q_target: y_stack} 
        return self.session.run([self.loss, self.train], feed_dict=feed)
    


