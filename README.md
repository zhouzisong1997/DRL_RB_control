# DRL_RB_control

This code is primarily built upon deep reinforcement learning (DRL) to control turbulent Rayleigh-Bénard (R-B) convection, aiming to enhance heat transfer at the walls. 

The implementation consists of two main components:
1. The DRL module, developed in Python and built upon the open-source code TurbulenceControlCode (GitHub: https://github.com/taehyuklee/TurbulenceControlCode  Related paper: [Lee et al (2023)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.8.024604?ft)).
2. The direct numerical simulation (DNS) module, implemented in Fortran and based on the open-source code AFiD (GitHub: https://github.com/PhysicsofFluids/AFiD  Related paper: [van der Poel et al (2015)](http://dx.doi.org/10.1016/j.compfluid.2015.04.007)).

We gratefully acknowledge the original authors of these open-source codes for their immense contributions to the current implementation! 

Detailed architecture and computational parameters are documented in the accompanying paper:
Deep reinforcement learning control unlocks enhanced heat transfer in turbulent convection (currently under review).

-----------------------------------------------------------------------------
### Usage Instructions

#### 1. Code Execution Environment
Both open-source code environments are required. Refer to their respective links for setup.

DNS Module Dependencies:
MPI, BLAS, LAPACK, FFTW3, HDF5 with parallel I/O.

DRL Module Dependencies:
Anaconda. 

The required Python libraries could be installed using the tfrl-cookbook.yml file from
TensorFlow 2 Reinforcement Learning Cookbook
(GitHub: https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook).

#### 2. Code Compile DNS Module
Navigate to the 'DNS_code' directory. The compilation process mirrors the open-source AFiD code.
Recommended command:
```
autoreconf -i
./configure
make
```

#### 3. Configure DNS Output Directory

The 'DNS_result' folder stores simulation data.

Replace the afid executable in this directory with the recompiled afid from the DNS module.

The initial subfolder contains: Initial field data for Ra=1E7, and DNS computational parameters







#### Control of turublent flow through Deep Reinforcement Learning (Test Code)

Welcome to my repository

This repository contains the codes mentioned in the paper <span style="color:red"> "Turbulence Control for Drag Reduction through deep reinforcement learning" </span> (https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.8.024604?ft).


This repository has three directories (packages) containing three DRL model files.


DoubleQdu - only streamwise wall shear stress field data is used to train the model.

DoubleQdudw - Both streamwise and spanwise wall shear stress field data are utilized to train the model.

DoubleQdw - only spanwise wall shear stress field data is used to train the model.



Each package contains the following Python scripts.
1. Environment.py (Fluid simulator connected to DRL model)
2. actor net.py (actor network)
3. critic net.py (critic network)
4. main xx.py (main DRL algorithm)
5. monitoring.py (it is for monitoring of fluid behavior in learning and statistics for turbulence)
6. ou noise.py (we formerly utilized the ou noise.py script, but we no longer do so you do not need to care about this file)

in addition, we used version 1 of TensorFlow

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
