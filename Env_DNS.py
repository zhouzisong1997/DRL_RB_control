#!/usr/bin/env python
# Environment: Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy


import numpy as np
import sys
import os
import random as random
import time
import math
import csv
import subprocess
import h5py




class env():
    def __init__(self, training_flag=True):

        print("--- CFD env init ---")

        nx_state = 256
        nz_state = 256
        nstate   = 1   # number of enviorment states 256*256*1
        nx_action = 256
        nz_action = 256
        naction = 1   # number of output actions 256*256*1

        nb_actuations0 = 1000*20 
        nb_actuations = nb_actuations0-1


        params={'nx_states': nx_state,
            'nz_states': nz_state,
            'nstates': nstate,
            'nx_actions': nx_action,
            'nz_actions': nz_action,
            'nactions': naction,
            'nactuations':nb_actuations,
            'min_actions': -1,
            'max_actions': 1,
            'nb_actuations':nb_actuations,
            'simulationfolder':'DNS_result',
            'ncpu':1,
            'action_file':'T_action.h5',
            'cdcl_file':'nu_plate.out',
            'state_file1':'T_state_low.h5',  
            'Model_save':2,
            'Model_begin':0,
            'Penalty_energy':50.0,
            'TrainFlag':training_flag,
            'nb_actuations0':nb_actuations0,
        }  

        self.params = params
        self.Nu1 = 0.0
        self.Nu2 = 0.0
        self.ncrash = 0
        self.probes_values = np.zeros((self.params['nx_states'],self.params['nz_states'],self.params['nstates']))
        self.state = self.probes_values
        self.dragreward = 0.0
        self.reward = 0.0
        self.alpha_output = 0.0
        self.episode_number = 0

        self.previous_modes_number = 0

        self.run_hist='train_hist.plt'

        
        self.start_class(complete_reset=True)

        self.done = False




        # directory for local SSD
        local_tmpdir = os.environ['LOCAL_TMPDIR']
        with open('DNS_result/tmpdir.txt', 'w') as file:
            file.write(local_tmpdir + "\n")





        print("--- CFD env init done! ---")

    def start_class(self, complete_reset=True):
        self.episode_number = 0
        self.action_number  = 0
        
        if self.params['TrainFlag']:
            # If it is a training run
            f = open(self.run_hist,'a+')
            f.close()

        else:
            # Otherwise
            f = open(self.run_hist,'a+')
            f.close()
        
        self.ready_to_use = True

    
    def initialize_flow(self, complete_reset=True):

        if complete_reset:

            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/T_action_old_initial.h5','T_action_total.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/bou.in','bou.in'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/cordin_info_initial.h5','cordin_info.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_master_initial.h5','continua_master.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vx_initial.h5','continua_vx.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vy_initial.h5','continua_vy.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vz_initial.h5','continua_vz.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_temp_initial.h5','continua_temp.h5'))

            self.exeCFD()


            a11 = True 
            while a11:
                new_folder_path = os.environ['LOCAL_TMPDIR']
                file_path = os.path.join(new_folder_path, 'continua_py.dat')
                file_path1 = os.path.join(new_folder_path, self.params['state_file1'])

                if os.path.exists(file_path):
                    with h5py.File(file_path1, 'r') as f1:
                        dataset = f1['state'] 
                        cps1 = np.array(dataset)
                    os.remove(file_path)
                    a11 = False
                else:
                    a11 = True


            cps = np.zeros((self.params['nx_states'],self.params['nz_states'],1))
            cps2 = cps1.reshape(self.params['nx_states'],self.params['nz_states'])
            

            cps[:, :, 0] = cps2.copy()

            self.probes_values = cps.copy()
            self.state = self.probes_values
            self.real_actions  = np.zeros((self.params['nx_actions'],self.params['nz_actions'],self.params['nactions']))
            self.last_actions  = np.zeros((self.params['nx_actions'],self.params['nz_actions'],self.params['nactions']))
            self.snapshots=[]




        else:
            pass

    def reset(self):

        #Reset environment and setup for new episode.
        #Returns:initial state of reset environment.

     
        self.initialize_flow(complete_reset=True)

        self.state = self.probes_values
        self.episode_number += 1

        self.action_number  = 0


        return self.state    



    def reset_test(self):

        #Reset environment for the testing process

        self.initialize_flow_test(complete_reset=True)


        self.state = self.probes_values
        self.episode_number = 0
        self.action_number  = 0


        self.run_hist='test_hist.plt'
        f = open(self.run_hist,'a')
        f.close()



        return self.state 



    def initialize_flow_test(self, complete_reset=True):

        if complete_reset:

            self.previous_modes_number = 0


            dest_file = os.path.join(self.params['simulationfolder'], 'initial/continua_master_initial_1.h5')
            src_file = os.path.join(self.params['simulationfolder'], 'initial/continua_master_initial.h5')
            if not os.path.exists(dest_file):
                os.system('cp {} {}'.format(src_file, dest_file))

            dest_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vx_initial_1.h5')
            src_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vx_initial.h5')
            if not os.path.exists(dest_file):
                os.system('cp {} {}'.format(src_file, dest_file))

            dest_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vy_initial_1.h5')
            src_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vy_initial.h5')
            if not os.path.exists(dest_file):
                os.system('cp {} {}'.format(src_file, dest_file))

            dest_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vz_initial_1.h5')
            src_file = os.path.join(self.params['simulationfolder'], 'initial/continua_vz_initial.h5')
            if not os.path.exists(dest_file):
                os.system('cp {} {}'.format(src_file, dest_file))

            dest_file = os.path.join(self.params['simulationfolder'], 'initial/continua_temp_initial_1.h5')
            src_file = os.path.join(self.params['simulationfolder'], 'initial/continua_temp_initial.h5')
            if not os.path.exists(dest_file):
                os.system('cp {} {}'.format(src_file, dest_file))



            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/T_action_old_initial.h5','T_action_total.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/bou_test.in','bou.in'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/cordin_info_initial.h5','cordin_info.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_master_initial_1.h5','continua_master.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vx_initial_1.h5','continua_vx.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vy_initial_1.h5','continua_vy.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_vz_initial_1.h5','continua_vz.h5'))
            os.system('cp {0}/{1} {0}/{2}'.format(self.params['simulationfolder'],'initial/continua_temp_initial_1.h5','continua_temp.h5'))

            self.exeCFD()

            a11 = True 
            while a11:
                new_folder_path = os.environ['LOCAL_TMPDIR']
                file_path = os.path.join(new_folder_path, 'continua_py.dat')
                file_path1 = os.path.join(new_folder_path, self.params['state_file1'])

                if os.path.exists(file_path):
                    with h5py.File(file_path1, 'r') as f1:
                        dataset = f1['state'] 
                        cps1 = np.array(dataset)
                    os.remove(file_path)
                    a11 = False
                else:
                    a11 = True

            cps = np.zeros((self.params['nx_states'],self.params['nz_states'],1))
            cps2 = cps1.reshape(self.params['nx_states'],self.params['nz_states'])


            cps[:, :, 0] = cps2.copy()

            self.probes_values = cps.copy()
            self.state = self.probes_values
            self.real_actions  = np.zeros((self.params['nx_actions'],self.params['nz_actions'],self.params['nactions']))
            self.last_actions  = np.zeros((self.params['nx_actions'],self.params['nz_actions'],self.params['nactions']))
            self.snapshots=[]




        else:
            pass








    def writeActions(self,actions):

        # This function is to write the parameters that will be used in the CFD simulations.

        self.last_actions = self.real_actions
        self.real_actions = actions


        actions0 = actions.flatten()

        new_folder_path = os.environ['LOCAL_TMPDIR']
        hdf5_file = os.path.join(new_folder_path, self.params['action_file'])

        with h5py.File(hdf5_file, "w") as f2:
            dataset = f2.create_dataset("action", actions0.shape, dtype=np.float64)
            dataset[...] = actions0


        new_folder_path = os.environ['LOCAL_TMPDIR']
        drag_file = os.path.join(new_folder_path, self.params['cdcl_file'])


        os.system('rm {0}'.format(drag_file))


        new_folder_path = os.environ['LOCAL_TMPDIR']
        con_file = os.path.join(new_folder_path, 'continua.dat')

        f = open(con_file,'w')
        f.close()





    def exeCFD(self):

        # Run CFD simulations

        os.chdir(self.params['simulationfolder'])

        # The directory needs to be adjusted based on the location of DNS code's executable file!
        os.system('mpirun -np 256 /control_DRL/code-1e7-0/DNS_result/afid &')


        os.chdir('..')






    def compute_Nu(self,Nu_low,Nu_upp):


        dragr = Nu_low/17.114-1.0
        dragr = 10.0*dragr

        return dragr
    

        
    def readEnv(self):

        # This function is to read the results of the CFD simulation.
        # These results may include some pressures/velocity and lift/drag coefficients.
        # We will use them to perceive the environments and also to compute the reward.


        a11 = True 
        while a11:
            try:
                flagnan = False
                new_folder_path = os.environ['LOCAL_TMPDIR']
                drag_file = os.path.join(new_folder_path, self.params['cdcl_file'])
                cdcl = np.loadtxt(drag_file)
                if cdcl.size == 0:
                    a11 = True
                else:
                    break
            except:
                a11 = True


        


        if not flagnan:

            cdcl1=cdcl.reshape(-1, 3)

            Nu_low = cdcl1[-1,1]
            Nu_upp = cdcl1[-1,2]

            self.dragreward = self.compute_Nu(Nu_low,Nu_upp)

            self.Nu1 = Nu_low
            self.Nu2 = Nu_upp



            a11 = True 
            while a11:
                new_folder_path = os.environ['LOCAL_TMPDIR']
                file_path = os.path.join(new_folder_path, 'continua_py.dat')
                file_path1 = os.path.join(new_folder_path, self.params['state_file1'])

                if os.path.exists(file_path):
                    with h5py.File(file_path1, 'r') as f1:
                        dataset = f1['state'] 
                        cps1 = np.array(dataset)
                    os.remove(file_path)
                    a11 = False
                else:
                    a11 = True


            cps = np.zeros((self.params['nx_states'],self.params['nz_states'],1))
            cps2 = cps1.reshape(self.params['nx_states'],self.params['nz_states'])

            cps[:, :, 0] = cps2.copy()

            self.probes_values = cps.copy()

        else:
            print("CFD simulation didn't converge! Terminated.")
            sys.exit()    

        terminal = False
        f = open(self.run_hist,'a+')
        if self.params['TrainFlag']:
            mytime = self.previous_modes_number + self.action_number//(self.params['nactuations']+1) + \
                    (self.action_number%((self.params['nactuations']+1)))/(self.params['nactuations']+1) + self.episode_number-1
        else:
            mytime = self.previous_modes_number + self.action_number//(self.params['nactuations']+1) + \
                    (self.action_number%((self.params['nactuations']+1)))/(self.params['nactuations']+1) + self.episode_number
        f.write('%.15f %.15f %.15f %.15f\n'%(mytime,self.Nu1,self.Nu2,self.dragreward))
        

        return terminal   

    
       



    def compute_reward(self):

        return self.dragreward






        
    def step(self, actions=None, episode00=0):

        self.previous_modes_number = episode00

        if actions is None:
            actions = np.zeros((self.params['nx_actions'],self.params['nz_actions'],self.params['nactions']))
        


        #Write actions for CFD simulations
        self.writeActions(actions)

        #Read CFD results
        terminal = self.readEnv()
        next_state = self.probes_values
        reward = self.compute_reward()
        self.reward = reward
        self.action_number += 1

        self.done = False
        if self.action_number >= (self.params['nactuations']+1):
            self.done = True


        self.state = self.probes_values

        return next_state, reward, self.done
    



 


