#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import tensorflow.compat.v1 as tf
import random
from collections import deque
from ou_noise import OUNoise
import critic_net
import actor_net
import os
import Env_DNS as En
import h5py

State_Episode_path = 'State & episode_unit_record/'
if not os.path.exists(State_Episode_path):
    os.makedirs(State_Episode_path)

tf.compat.v1.disable_eager_execution()

environment = En.env() #call environment

alpha_critic = 1.0 #learning rate (based on Q)

#Input & Output
state_shape = (environment.params['nz_states'], environment.params['nx_states'], environment.params['nstates'])
action_shape = (environment.params['nz_actions'], environment.params['nx_actions'], environment.params['nactions'])



input_size_critic_s = state_shape
input_size_critic_a = action_shape


output_size_critic = 1 

input_size_actor_s = state_shape 
output_size_actor =  action_shape 


state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
action = np.zeros([action_shape[0], action_shape[1], action_shape[2]], dtype = np.float64)
step_decay = tf.placeholder(tf.float32, [1])
episode_decay = tf.placeholder(tf.float32, [1])
alpha = tf.placeholder(tf.float32, [1])



#Reinforcement learning parmeter
n_index =5
dis = 0.95
dis_target = dis**n_index
buffer_memory = 30000 #Replay memory (Buffer)
exploration_noise = OUNoise(action_shape) #OU noise
del_distance = 0.03
clip = 0.3

def n_step(Temporal_stack, state_step, n_index):

    global dis

    f_list = Temporal_stack[state_step-n_index]

    n_ind_list = Temporal_stack[state_step]

    reward_avg  = 0.0
    dis_index = 0

    for _ in range ((state_step-n_index), (state_step)):
        batch = Temporal_stack[_]
        reward_steps = batch[2]*(dis**dis_index)

        reward_avg += reward_steps/n_index
        dis_index +=1

    state = f_list[0]; action_noise = f_list[1]; reward =reward_avg ; next_state = n_ind_list[3]; end = n_ind_list[4]

    return state, action_noise, reward, next_state, end


def critic_train(main_critic_1, target_critic_1, main_critic_2, target_critic_2, main_actor, target_actor, train_batch):
    Q_old = np.zeros([1], dtype = np.float64) 
    Q_new = np.zeros([1], dtype = np.float64)
    
    x_action_stack = np.empty(0)
    x_state_stack = np.empty(0)
    y_stack = np.empty(0)
    
    x_state_stack = np.reshape(x_state_stack, (0, state_shape[0]*state_shape[1]*state_shape[2]))  
    x_action_stack = np.reshape(x_action_stack, (0, action_shape[0]*action_shape[1]*action_shape[2]))
    y_stack = np.reshape(y_stack, (0, output_size_critic)) 

    for state, action, reward, next_state, end in train_batch: 

        scale_action = 0.2*np.std(action)

        #noise for action
        rand_num = np.random.normal(loc=0.0, scale=scale_action, size=(action_shape[0], action_shape[1], 1)) 
        noisy_act = np.clip(rand_num, -clip, clip ) 
     

        #----------------------------------------- next_action + noise -------------------------------------------#
        next_action = target_actor.predict(next_state[:]) 
        next_action_reshape = np.reshape(next_action, [action_shape[0], action_shape[1], action_shape[2]])
        next_act_noisy_2D = next_action_reshape[:,:,:] + noisy_act[:,:,:] 
        next_act_noisy = np.reshape(next_act_noisy_2D, [1, action_shape[0]*action_shape[1]*action_shape[2]])
        #---------------------------------------------------------------------------------------------------------#

        Q_new_1 = reward + dis_target*(target_critic_1.predict(next_state[:], next_act_noisy[:])) #target
        Q_new_2 = reward + dis_target*(target_critic_2.predict(next_state[:], next_act_noisy[:]))   
      

        Q_new = min(Q_new_1[0,0], Q_new_2[0,0])
        Q_new = np.reshape(Q_new, (1,1))

        y_stack = np.vstack([y_stack, Q_new])
        x_state_stack = np.vstack([x_state_stack, state]) 

        x_action_stack = np.vstack([x_action_stack, action])
        
    loss_critic_1, _ = main_critic_1.update(x_state_stack, x_action_stack, y_stack) 
    loss_critic_2, _ = main_critic_2.update(x_state_stack, x_action_stack, y_stack)
        
    return loss_critic_1, loss_critic_2, Q_old, Q_new



def actor_train(main_actor, noise_actor, main_critic, train_batch, coef_alpha, batch_size, sess):
    
    x_stack_actor = np.empty(0)
    x_stack_actor = np.reshape(x_stack_actor, (0, input_size_actor_s[0]*input_size_actor_s[1]*input_size_actor_s[2]))
    square = np.zeros([1], dtype=np.float64)
    
    for state, action, reward, next_state, end in train_batch: 
        
        imediate = np.zeros([1], dtype=np.float64)
        imediate = (main_actor.predict(state) - noise_actor.predict(state))**2
        square = imediate/batch_size + square

        x_stack_actor = np.vstack([x_stack_actor, state]) 

    distance = np.sqrt(np.mean(square))

    if distance < del_distance:
        coef_alpha *= 1.01
    else:
        coef_alpha /= 1.01

    _ = main_actor.update(main_critic, x_stack_actor)
        
        
    return coef_alpha, distance   


def first_copy (sess, target_scope_name ="target", main_scope_name = "main"):

    op_holder = []
    
    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

    for main_var, target_var in zip(main_vars, target_vars): 
        
        op_holder.append(target_var.assign(main_var.value()))

    return sess.run(op_holder)



def copy_var_ops(*, target_scope_name ="target", main_scope_name = "main"):

    op_holder = []
    tau = 0.001
    
    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

    for main_var, target_var in zip(main_vars, target_vars): 
        
        op_holder.append(target_var.assign(tau * main_var.value() +(1 - tau)*target_var.value()))

            
        
    return op_holder, main_var.value(), target_var.value()



def space_noise(noise_vars, noise_name ="noise_actor", main_name = "main_actor"):

    noise_stack = []
    noise_added_stack = []

    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_name)
    noise_added_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=noise_name)

    for main_var, noise_var in zip(main_vars, noise_vars): 
               
        noise_stack.append(noise_var.assign(noise_var + 0.2* (0 - noise_var) + tf.random_normal(tf.shape(main_var), mean = 0.0, stddev = (alpha)*tf.math.reduce_std(main_var),dtype=tf.float32)))

    for main_var, noise_var, noise_added_var in zip(main_vars, noise_vars, noise_added_vars): 
        noise_added_stack.append(noise_added_var.assign(main_var + noise_var))
  
    return noise_stack, noise_added_stack


def get_noise_var():  
    vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "main_actor")
    noise_vars = [tf.Variable(tf.zeros(var.shape,dtype=tf.float32),dtype=tf.float32) for var in vars1]
    return noise_vars 

def main():

    global n_index
    
    Q_old = np.empty(0)
    Q_new = np.empty(0)  
    Temporal_stack = []
    state_step = 0
    Loss_step = 0
    main_update_freq = 1
    actor_update_freq = 5
    target_update_frequency = 5
    train_loop_epoch = 1
    max_episodes = 100
    batch_size = 64 #Mini batch size Buffer
    buff_len = batch_size    
    temp_1=0 
    total_time = 0
    coef_alpha = 0.1
    append_t = False



    #------------ state time interval ------------#
    st_step = 1000
    step_deadline = environment.params['nb_actuations0'] 
    starting_act = batch_size*st_step
    noise_actor_freq = 5
    #---------------------------------------------#
    

    # Replay buffer
    buffer = deque(maxlen =buffer_memory) 
 

    config = tf.ConfigProto(log_device_placement=True)	
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
     
        #formation of network for actor net
        main_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="main_actor") 
        target_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="target_actor")  
        noise_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="noise_actor")  
       
        #formation of network for critic net 
        main_critic_1 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, main_actor, name="main_critic_1") 
        main_critic_2 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, main_actor, name="main_critic_2") 
        target_critic_1 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, target_actor, name="target_critic_1")    
        target_critic_2 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, target_actor, name="target_critic_2")   

        
        noise_vars = get_noise_var() 
        
        _ = main_actor.initialization_a(main_critic_1.Objective, name ="main_actor")
        _ = target_actor.initialization_a(target_critic_1.Objective, name ="target_actor")
        _ = main_critic_1.initialization_c(name ="main_critic_1")
        _ = main_critic_2.initialization_c(name ="main_critic_2")
        _ = target_critic_1.initialization_c(name ="target_critic_1")
        _ = target_critic_2.initialization_c(name ="target_critic_2")     



        saver_act = tf.train.Saver(max_to_keep=None) 

        checkpoint = tf.train.get_checkpoint_state('./Save_check/')
        if checkpoint and checkpoint.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver_act.restore(sess, checkpoint.model_checkpoint_path)
            checkpoint_path = checkpoint.model_checkpoint_path
            parts = checkpoint_path.split() 
            episode_number = parts[-1]  
            episode_number = int(episode_number)+1
        else:
            sess.run(tf.global_variables_initializer())
            episode_number = 0
            #Critic (first_copy)
            _ = first_copy(sess, target_scope_name="target_critic_1",main_scope_name="main_critic_1")
            _ = first_copy(sess, target_scope_name="target_critic_2",main_scope_name="main_critic_2")
            #Policy (first_copy)
            _ = first_copy(sess, target_scope_name="target_actor", main_scope_name="main_actor")



        print("initialization complete")
        

        
        #Critic (Copy)
        copy_critic_1, main_val_c, target_val_c = copy_var_ops(target_scope_name="target_critic_1",main_scope_name="main_critic_1")
        copy_critic_2, main_val_c, target_val_c = copy_var_ops(target_scope_name="target_critic_2",main_scope_name="main_critic_2")
        
        #Policy (Copy)
        copy_actor, main_val_a, target_val_a =  copy_var_ops(target_scope_name="target_actor",main_scope_name="main_actor")

        #Noise 
        noise_copy, noise_added_copy = space_noise(noise_vars, noise_name ="noise_actor", main_name = "main_actor")


        for episode in range(episode_number, max_episodes+1):
            
            
            print("Episode : {} start ".format(episode))    
            time = 0 
            end  = False
            state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
            next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
            action = np.zeros([action_shape[0], action_shape[1], action_shape[2]], dtype = np.float64)


            state_stock = environment.reset()
            state_stock = state_stock.swapaxes(0,1)


            exploration_noise.reset()

            state[:,:,0] = state_stock[:,:,0] 
 
            state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]]) 


            reward_graph = 0

            reward_record = open(State_Episode_path + "/reward.plt" , 'a', encoding='utf-8', newline='') 
            if temp_1 ==0: reward_record.write('VARIABLES = "Episode", "Reward" \n') 

 

            temp_1=1   

            noise_record = open(State_Episode_path + "/noise, episode{}.plt" .format(episode), 'a', encoding='utf-8', newline='')
            noise_record.write('VARIABLES = "state_step", "noise" \n')  


            while not end == True: 
                
                state_reward_record = open(State_Episode_path + "/state_reward.plt", 'a', encoding='utf-8', newline='')
                Loss_record = open(State_Episode_path + "/Loss_record.plt", 'a', encoding='utf-8', newline='')
                alpha_value = open(State_Episode_path + "/alpha_value (state unit).plt" , 'a', encoding='utf-8', newline='')                

                if episode == 0 and state_step == 0:
                    Loss_record.write('VARIABLES = "state", "Loss" \n')  

                if state_step == 0: 
                    state_reward_record.write('VARIABLES = "state_step", "avg_reward" \n') 
                    alpha_value.write('VARIABLES = "state_step", "coef_alpha" "distance"\n') 
                    

                if state_step % noise_actor_freq == 0:
                    state_step_f =  np.reshape(state_step, (1)) 
                    coef_f = np.reshape(coef_alpha, (1))
                    episode_f = np.reshape(episode, (1))
                    feed = {step_decay: state_step_f, episode_decay: episode_f, alpha: coef_f} 
                    sess.run([noise_copy], feed_dict=feed)
                    sess.run([noise_added_copy], feed_dict=feed)
                    total_time = total_time + 1 
 
                next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
                
                if append_t == True:
                    if len(buffer) > buff_len and state_step % main_update_freq == 0: 
                        loss_avg = 0
                    
                        for _ in range(train_loop_epoch):
                            minibatch = random.sample(buffer, batch_size) 
                            minibatch = list(minibatch)
                        
                            loss_critic, _ , Q_old, Q_new= critic_train(main_critic_1, target_critic_1, main_critic_2, target_critic_2, main_actor, target_actor, minibatch)
                        
                            if state_step % actor_update_freq ==0:
                                coef_alpha, distance  = actor_train(main_actor, noise_actor, main_critic_1, minibatch, coef_alpha, batch_size, sess)
                                alpha_value.write("%d %f %f \n" %(state_step, coef_alpha, distance))
                                alpha_value.close()

                            loss_avg = loss_critic/train_loop_epoch +loss_avg

                        print("Loss for critic is : {}".format(loss_avg))   
                        Loss_record.write("%d %f \n" %(Loss_step , loss_avg))
                        Loss_record.close()

                    if state_step > 1 and state_step % target_update_frequency == 0:
                        sess.run(copy_critic_1)
                        sess.run(copy_critic_2)
                    

                    if state_step > 1 and state_step % target_update_frequency == 0:
                        sess.run(copy_actor)
                        if episode == 0 and state_step < buff_len:
                            pass
                        else:
                            print("target update")

                
                
                Noise = 0*exploration_noise.noise() 
                Noise = Noise/((state_step*0.01+ episode*1+1))
                
                action = noise_actor.predict(state) + Noise

                action_noise = np.reshape(action, (action_shape[0]*action_shape[1]*action_shape[2])) 

                noise_record.write("%d %f \n" %(state_step ,np.mean(Noise)))



                state00=state
                reward01=np.zeros(int(st_step))
                for t00 in range(int(st_step)):
                    action00 = noise_actor.predict(state00) + Noise
                    action00 = np.reshape(action00, (action_shape[0], action_shape[1], action_shape[2]))
                    action00 = action00.swapaxes(0,1)
                    next_state00, reward00, end = environment.step(action00,episode_number)
                    reward01[t00] = reward00
                    state00 = next_state00.swapaxes(0,1)
                next_state_stock = state00
                start_index0 = int(st_step) // 2
                reward = np.mean(reward01[start_index0:])

                next_state[:,:,0] = next_state_stock[:,:,0] 

                next_state = np.reshape(next_state, [1, state_shape[0]*state_shape[1]*state_shape[2]]) 

               


                ################ N-time Replay memory ##############

                state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]])

                stack_element = (state, action_noise, reward, next_state, end)
                Temporal_stack.append(stack_element) 

                ##############################

                if state_step >= (n_index):

                    state, action_noise, reward_p, next_state_p, end = n_step(Temporal_stack, state_step, n_index)
                    buffer.append((state, action_noise, reward_p, next_state_p, end))

                    append_t = True

                    reward_graph = reward_p + reward_graph

                    state_reward_record.write("%d %f \n" %(state_step ,reward_p))
                    state_reward_record.close() 

                    if len(buffer) > buffer_memory:
                        buffer.popleft()

                ################################################################       

               
                state = next_state
                state_step = state_step + 1
                time = time + 1
                Loss_step += 1



           
            
            os.system('pkill -f "afid"')
            
            saver_act.save(sess, './Save_check/model episode {}'.format(episode), global_step = None)





            reward_graph = reward_graph/state_step
            
            Temporal_stack = [] 

            reward_record.write("%d %f \n" %(episode , reward_graph))
            

            noise_record.close()
            
            state_step = 0
            
            reward_record.close()  



            



        #---------------------------------------------------------- Test part ---------------------------------------------------------#
                
        end_step = 400000

        state_stock = environment.reset_test() #reset environment
        state_stock = state_stock.swapaxes(0,1)
        state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
        state[:,:,0] = state_stock[:,:,0]
        state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]])


        try:
            with open("episode_test.txt", "r") as file:
                episode0 = int(file.read())
        except FileNotFoundError:
            with open("episode_test.txt", "w") as file:
                file.write("0")
            episode0 = 0
        
        

        state00=state
        reward02=np.zeros(int(end_step))
        for t00 in range(int(end_step)):
            action00 = main_actor.predict(state00)
            action00 = np.reshape(action00, (action_shape[0], action_shape[1], action_shape[2]))
            action00 = action00.swapaxes(0,1)
            next_state00, reward00, end = environment.step(action00,episode0)
            reward02[t00] = reward00
            state00 = next_state00.swapaxes(0,1)

            if (t00-100) % environment.params['nb_actuations0'] == 0:
                with open("episode_test.txt", "w") as file:
                    file.write(str((t00-100) // environment.params['nb_actuations0'] + episode0))

                    dest_file = os.path.join(environment.params['simulationfolder'], 'initial/continua_master_initial_1.h5')
                    src_file = os.path.join(environment.params['simulationfolder'], 'continua_master.h5')
                    os.system('cp {} {}'.format(src_file, dest_file))
                    dest_file = os.path.join(environment.params['simulationfolder'], 'initial/continua_vx_initial_1.h5')
                    src_file = os.path.join(environment.params['simulationfolder'], 'continua_vx.h5')
                    os.system('cp {} {}'.format(src_file, dest_file))
                    dest_file = os.path.join(environment.params['simulationfolder'], 'initial/continua_vy_initial_1.h5')
                    src_file = os.path.join(environment.params['simulationfolder'], 'continua_vy.h5')
                    os.system('cp {} {}'.format(src_file, dest_file))
                    dest_file = os.path.join(environment.params['simulationfolder'], 'initial/continua_vz_initial_1.h5')
                    src_file = os.path.join(environment.params['simulationfolder'], 'continua_vz.h5')
                    os.system('cp {} {}'.format(src_file, dest_file))
                    dest_file = os.path.join(environment.params['simulationfolder'], 'initial/continua_temp_initial_1.h5')
                    src_file = os.path.join(environment.params['simulationfolder'], 'continua_temp.h5')
                    os.system('cp {} {}'.format(src_file, dest_file))



        reward = np.mean(reward02[:])



        print("Test is finished")
        #------------------------------------------------------------------------------------------------------------------------------#
            

if __name__ == "__main__":

    main()

    print("All process is finished!")