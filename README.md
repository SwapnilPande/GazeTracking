# EyeTracking
## Objective
Accurately estimate gaze direction of a user using inexpensive, readily-available cameras

## Application/Rationale
Gaze-based navigation control for powered wheelchair users

## Performance Objectives/Requirements:
1. Shall be able to track gaze of all users with various eye structures (large vs. small eyes, covering of iris, contacts, glasses, etc.)
2. Shall be capable of tracking gaze in common lighting settings (bright, dim, outdoors, various color temperatures)
3. Shall use only inexpensive, readily-available cameras and no additional sensors to track eyes/gaze
4. Shall be capable of tracking gaze at "acceptable" rate on inexpensive hardware capable of being attached to powered wheelchairs (SBC such as RPi)
5. Shall be unobstrusive to user
6. Shall achieve "acceptable" accuracy

## Deliverables and Desired Outcomes:
1. Software package capable of gaze estimation meeting defined performance requirements
    * Shall return a gaze vector of tracked user
2. Defined operating parameters for software package
    * Angular accuracy of gaze estimation
    * Max/min detection distance and angle (between sensor and user)
    * Calibration requirements
3. Infrastructure and pipeline for data collection and analysis
4. Machine learning infrastructure and environment
    * Allow for easy iteration for model training
    * Allow for rapid development for future projects
5. Open source library for gaze tracking data

## Project roadmap
1. Research eye-tracking and current solutions - Week 1 (6/18-6/22)
    * Study both ML and non-ML solutions
    * Determine performance parameters and limitations
2. Resarch Machine Learning - Week 1-2 (6/18-6/29)
    * Study current techniques and model architectures
    * Study current tools for building ML pipelines
        * Examples include Deepforge, Tensorflow, Caffe, Torch, Keras, etc.
3. Replicate performance of current state-of-the-art solutions Week 2-4 (6/25-7/13)
    * Krafka's [Eye tracking for everyone](http://gazecapture.csail.mit.edu/) (Krafka et al.)
    * Build testing environment and evaluate performance
5. Build codebase for "plug-n-play" ML development Week 4 - (7/9-7/13)
    * Aim to minimize work required to train and implement various ML models
4. Build new ML Model Week 4-7 (7/9-9/3)
    * Ignore runtime performance optimization
    * Evaluate performance in testing environment
    * Iterate and re-evaluate
5. Evaluate performance and progress, build continued roadmap
    
        
