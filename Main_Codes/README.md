
# Pianoptim

## Introduction

**Pianoptim** is a research project focused on optimizing piano technique by evaluating the biomechanical effects of proximal playing strategies. Playing-related musculoskeletal disorders (PRMDs) are common among pianists, posing significant challenges to their health and performance capabilities. This study utilizes optimal control theory to minimize distal joint torques, where common PRMDs are found, through the use of dynamic and static trunk playing strategies during simple piano tasks.

### Keywords
- Piano performance
- Kinematics
- Musculoskeletal disorders
- Optimal Control theory
- Trunk Motion

### Project Description

We developed a 3D torque-driven upper body biomechanical model with 12 degrees of freedom (DOF) to analyze piano playing techniques. The model, implemented using the biorbd_casadi framework, integrates musculoskeletal modeling and optimal control.

The focus is on reducing torque loads on distal joints, which are prone to PRMDs. The model simulates two types of piano touch—struck and pressed—using Dynamic Trunk (DT) and Static Trunk (ST) strategies. DT involves active trunk movement to distribute biomechanical load and reduce torque on wrists and fingers, while ST keeps the trunk stationary, increasing torque on distal joints.

The simulation covers five phases: Preparation, Key Descent, Key Bed, Key Release, and Return to Neutral. We used the bioptim library to define and solve an optimal control program (OCP) with constraints and objectives tailored to realistic piano playing movements.

### Folder Descriptions
   
    UDEM FILES
        EKSAP research day: Includes a poster and a one-page abstract about the project.
        Energy transferring: Contains articles on calculating or finding the flow of energy from proximal to distal joints.
        Literature review: Contains articles and a PowerPoint presentation worth checking out.
        Scholarship: Materials related to two scholarship applications and notes from a workshop on writing good scholarship applications.
        Seminar course: Presentation about the project and related notes.
        EKSAP committee: First-year presentation (optional to review).
        
    Calibration_Process_Force_Sensor
    
        Methodology adopted to calibrate the sensors for measuring force profiles.
    
    Geometry
    
    Mesh files
        Includes all the mesh files (not needed to check as they are just used for visualization).
        
    updated biomod ifle yeadon model
    wu biomod file
    
        Previously, we used the Wu model and at some point, we decided to switch to the Yeadon model due to technical reasons. Here are the necessary files:
        Includes conditions with and without the .biomod file, representing static (ST, without) and dynamic trunk (DT, with) conditions.
        Analysis of Tau reduction.py: Used for showing the result of comparing two conditions and the percentages of tau reduction.
        Biorbd_phase1_velocity.py: Shows the contribution of each joint in the velocity profile of the distal joints.
        Biorbd_phase2_DynamicEQ.py: Shows the contribution of each part of the dynamic equation in phase 2.
        Comparison Matplotlib and Seaborn: Files for comparing the kinetic and kinematic graphs/plots via Seaborn or Matplotlib libraries.
        DJ.py: About dimensionless jerk calculation, which you can adopt and check before use.
        Piano_distance_pianist.py: About the distance between the piano and the pianist.
        Pressed_struck.py and Pressed_struck_Updated_file_2.py: The main files for simulations and working on.
        Show animation: Shows animations with Bioviz.
        Work_joint and tau_theta: Calculates the work done by each joint, an idea that can be explored further (refer to the articles in the Literature folder for more information).



 
