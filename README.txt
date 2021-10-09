
==================================================================================================================

Types of file contained in the tar file:

    • Module files BMAP3D.py for the Body Frame Module; INPUT.py for the Input Module; EYES.py for the Eye Module; ARMS.py for the Arm Module; COLM.py for the Column Module; LEGS.py for the Leg Module; PTSS.py for the PTSS Module; FUNCS.py for the Function Module; for convenience, the corresponding part of the ERG Module is included in the Arm Module, Column Module and Leg Module.
    
    • C extension files CFNS.pyx, CPTS.pyx and their auxillary for speeding up computations;
    
    • Learning trace copies Eye2040.pickle, Arm2040.pickle, Col2040.pickle, Leg2040.pickle for updating the learning variables, where the resolution uses array size 40x40;
    
    • Animation files PAN01.py for set one, PAN02.py for set two, PAN03.py for set three, PAN04.py for set four, and PAN05.py for set five on the website;
    
    • Text file README.txt for instructions and descriptions of the files.
    
    • NB: The simulations were performed on a Fedora 33 Workstation with Intel Core i7-9750H CPU @ 2.60GHz × 12, 16GB RAM, 2TB Hard Drive. 
    
==================================================================================================================
    
Prerequisites for the animation files:

    • Python and its modules numpy, matplotlib, pickle, cypthon.

==================================================================================================================

Descriptions of the animation files:

    • In PAN01.py, the network makes reactive saccade to a sequence of targets and estimates the spatial position of each target.

    • In PAN02.py, the network makes independent head movements and arm movements according to a motor program of motor commands organized relative to the standard anatomical position.

    • In PAN03.py, the network performs locomotion under cortical and subcortical control, where each type of control is a form of the central pattern generator.

    • In PAN04.py, the network practices eye-hand coordination from side to side that is assisted by eye-head coordination.
    
    • In PAN05.py, the network practices locomotion by visual guidance that is assisted by eye-head coordination.
	

==================================================================================================================

Instructions for the animation files:

    • For PAN01.py, either run the file in the command line by “python PAN01.py” or in an IDE.

    • For PAN02.py, either run the file in the command line by "python PAN02.py" or in an IDE.
      
    • For PAN03.py, the default option is “Locomotion with cortical and subcortical CPGs”.
    
        ◦ For option “Locomotion with subcortical CPG”, the adjustments are 
            ▪ the user need to comment out methods “Leg.ParietalSpat()” and “Leg.Motor()” in PAN03.py to turn off these brain regions;
            ▪ the user need to use the first block of code in the “SpinalCore()” method in LEGS.py but comment out the second and third block of code which are for the other options;
            ▪ then either run the file in the command line by “python PAN03.py” or in an IDE.
             
        ◦ For option “Locomotion with cortical CPG”, the adjustments are
            ▪ the user need to comment out method “Leg.Brainstem()” in PAN03.py to turn off this brain region;
            ▪ the user need to use the second block of code in the “SpinalCore()” method in LEGS.py, but comment out the first and third block of code which are for the other options;
            ▪ then either run the file in the command line by “python PAN03.py” or in an IDE.

    • For PAN04.py, the adjustments are
    
        ◦ the user need to use the second block of code in the “ParietalSpat()” method of ARMS.py to read out the learned motor commands, but comment out the third block of code to turn off the corresponding part of the ERG Module in this file;
        ◦ the user need to use the second block of code in the “SpinalCore()” method in ARMS.py, but comment out the first block of code which is for file PAN02.py;
        ◦ then either run the file in the command line by “python PAN04.py” or in an IDE.


    • For PAN05.py, the adjustments are
    
        ◦ the user need to use the second block of code in the “ParietalSpat()” method of LEGS.py to read out the learned motor commands, but comment out the third block of code to turn off the corresponding part of the ERG Module in this file;
        ◦ the user need to use the fourth block of code in the “SpinalCore()” method in LEGS.py, but comment out the first, second and third block of code which is for file PAN03.py;
        ◦ then either run the file in the command line by “python PAN05.py” or in an IDE.


    • NB: Give each animation a few minutes to build up the network activity.
     
==================================================================================================================



