# DRL_RB_control

This code is primarily built upon deep reinforcement learning (DRL) to control turbulent Rayleigh-Bénard (R-B) convection, aiming to enhance heat transfer at the walls. 

The implementation consists of two main components:
1. The DRL module, developed in Python and built upon the open-source code TurbulenceControlCode
   (https://github.com/taehyuklee/TurbulenceControlCode  Related paper: [Lee et al (2023)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.8.024604?ft)).
2. The direct numerical simulation (DNS) module, implemented in Fortran and based on the open-source code AFiD
   (https://github.com/PhysicsofFluids/AFiD  Related paper: [van der Poel et al (2015)](http://dx.doi.org/10.1016/j.compfluid.2015.04.007)).

We gratefully acknowledge the original authors of these open-source codes for their immense contributions to the current implementation! 

Detailed architecture and computational parameters are documented in the accompanying paper:
Deep reinforcement learning control unlocks enhanced heat transfer in turbulent convection (currently under review).

-----------------------------------------------------------------------------
## Usage Instructions

### 1. Code Execution Environment
Both open-source code environments are required. Refer to their respective links for setup.

DNS Module Dependencies:
MPI, BLAS, LAPACK, FFTW3, HDF5 with parallel I/O.

DRL Module Dependencies:
Anaconda. 

The required Python libraries could be installed using the tfrl-cookbook.yml file from
TensorFlow 2 Reinforcement Learning Cookbook
(https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook).

### 2. Code Compile DNS Module
Navigate to the 'DNS_code' directory. The compilation process mirrors the open-source AFiD code.
Recommended compile command:
```
autoreconf -i
./configure
make
```

### 3. Configure DNS Output Directory

The 'DNS_result' folder stores simulation data.

Replace the 'afid' executable in this directory with the recompiled afid from the DNS module.

The initial subfolder contains: Initial field data for Ra=1E7, and DNS computational parameters.


### 4. DRL-DNS File Interaction
Key DRL files in the main directory:
Env_DNS.py, main_dw.py, actor_net.py, critic_net.py, ou_noise.py

Env_DNS.py handles DRL-DNS file exchange.

The author uses the compute node's local SSD for I/O operations (directory specified by the 'local_tmpdir' variable).
If no local SSD is available in your environment, modify this configuration accordingly.

Adjust paths in the following section of Env_DNS.py to initialize the DNS process:
```
os.system('mpirun -np your_cores /your_directory/DNS_result/afid &')
# Sample
# os.system('mpirun -np 256 /control_DRL/code-1e7-0/DNS_result/afid &')
```

### 5.Modify DRL Hyperparameters
Customize hyperparameters in the DRL module as needed. The current version is designed for the R-B convection at Ra=1E7.

### 6.Finish
After verifying all modifications, execute the program using:
```
python main_dw.py >> output00.txt
```


### Thank you!
-----------------------------------------------------------------------------
Additionally, Each model package has the TestCode directory.

This directory provides meta data for trained weight and bias, as well as Test code.

Test.py provides environment and actor model configuration code for the drag reduction test.

And other Environment.py, actor.py, and critic.py are identical to files in the upper directory.


thank you


-----------------------------------------------------------------------------

<p align="center">
<img src="https://user-images.githubusercontent.com/89365465/235430125-e0d680cd-cbee-4c26-b01d-59a75b1e1354.gif" width="49%" height="49%">
<img src="https://user-images.githubusercontent.com/89365465/235430132-b2e5457c-395d-448c-ad62-4c0a7361d524.gif" width="49%" height="49%">
<figcaption align="center">No Control-shear flow & Controlled Shear flow</figcaption>
</p>
<!-- ![no control 360](https://user-images.githubusercontent.com/89365465/235430125-e0d680cd-cbee-4c26-b01d-59a75b1e1354.gif)-->
<!-- ![Controlling shear flow](https://user-images.githubusercontent.com/89365465/235430132-b2e5457c-395d-448c-ad62-4c0a7361d524.gif)-->
