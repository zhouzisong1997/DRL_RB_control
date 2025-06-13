#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf_2


class actor:

    def __init__(self, session, input_size_actor_s, output_size_actor, output_size_critic,  name="main"): 


        self.session = session
        self.input_size_actor_s = input_size_actor_s 
        self.output_size_actor = output_size_actor 
        self.net_name = name
        
        print(name)
        self.build_network() 
        


    def build_network(self, k_1size = 64, k_2size=32, k_3size=1, k_4size=1): 

        def periodic_pad(X, padding_size=1): 
            X = tf.concat([X[:,-padding_size:,:,:], X, X[:,0:padding_size,:,:]], axis=1)
            X = tf.concat([X[:,:,-padding_size:,:], X, X[:,:,0:padding_size,:]], axis=2)
            return X
    
        with tf.variable_scope(self.net_name, reuse = tf.AUTO_REUSE): 
            
            self.X_input = tf.placeholder(tf.float32, [None, self.input_size_actor_s[0]*self.input_size_actor_s[1]*self.input_size_actor_s[2]], name="X_input")
            self.X_input_2 = tf.reshape(self.X_input, [-1, self.input_size_actor_s[0], self.input_size_actor_s[1], self.input_size_actor_s[2]]) 
            
            
            # Convolution layer 1
            k1_S = [3, 3, self.input_size_actor_s[2], k_1size]
            k1_std = np.sqrt(2)/np.sqrt(np.prod(k1_S[:-1]))
            
            self.kernal_1 = tf.get_variable("W_a1", shape =[3 , 3, self.input_size_actor_s[2], k_1size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k1_std, seed=None))
            self.B_a1 = tf.get_variable("B_a1", shape = [1,1, 1, k_1size], initializer = tf.initializers.zeros())
            
            linear_1 = tf.nn.conv2d(periodic_pad(self.X_input_2), self.kernal_1, strides=[1, 1, 1, 1], padding='VALID') + self.B_a1
            active_1 = tf.nn.relu(linear_1)

            
            # Convolution layer 2
            k2_S = [3, 3, k_1size, k_2size]
            k2_std = np.sqrt(2)/np.sqrt(np.prod(k2_S[:-1]))
            
            self.kernal_2 = tf.get_variable("W_a2", shape =[3, 3, k_1size, k_2size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k2_std, seed=None))
            self.B_a2 = tf.get_variable("B_a2", shape = [1,1, 1, k_2size], initializer = tf.initializers.zeros())
            linear_2 = tf.nn.conv2d(periodic_pad(active_1), self.kernal_2, strides=[1, 1, 1, 1], padding='VALID') + self.B_a2
            active_2 = tf.nn.relu(linear_2)

            
            # Convolution layer 3
            k3_S = [3, 3, k_2size, k_3size]
            k3_std = np.sqrt(1)/np.sqrt(np.prod(k3_S[:-1]))
            
            self.kernal_3 = tf.get_variable("W_a3", shape =[3, 3, k_2size, k_3size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k3_std, seed=None))
            self.B_a3 = tf.get_variable("B_a3", shape = [1,1, 1, k_3size], initializer = tf.initializers.zeros())
            linear_3 = tf.nn.conv2d(periodic_pad(active_2), self.kernal_3, strides=[1, 1, 1, 1], padding='VALID') + self.B_a3

            
            linear_4 = tf.reshape(linear_3, [-1, self.output_size_actor[0]*self.output_size_actor[1]*self.output_size_actor[2]])

            output = linear_4 - tf.reduce_mean(linear_4, axis=1, keepdims=True) 
            self.action_pred = 0.1*output 

            

            
            print("Actor_net connected")
            
            
    def initialization_a (self, Objective, name ="ops_name", l_rate=0.001, B = 0.000001):
        
        self.regular = tf.nn.l2_loss(self.kernal_1) + tf.nn.l2_loss(self.kernal_2) +tf.nn.l2_loss(self.kernal_3) 
        
        self.Objective = Objective 

        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)
        
        self.train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(-self.Objective, var_list = self.actor_vars)
            

    def predict(self, state): 
        x = np.reshape(state, [1, self.input_size_actor_s[0]*self.input_size_actor_s[1]*self.input_size_actor_s[2]])       
        return self.session.run(self.action_pred, feed_dict={self.X_input: x})


    
    
    def update(self, critic_net, x_stack):
        feed = {self.X_input: x_stack, critic_net.input_critic_state: x_stack} 
           
        return self.session.run([self.train], feed_dict=feed)
    
