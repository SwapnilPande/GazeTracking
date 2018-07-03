# iTracker
Project directory for replicating iTracker Convolutional Neural Network outlined in [Eye tracking for everyone](http://gazecapture.csail.mit.edu/) (Krafka et al., 2016).

## Setup
### TensorFlow
TensorFlow is the machine learning package used to build and train the model. Follow the installation instructions found [here](https://www.tensorflow.org/install/) for your specific platform. 

NOTE: If you only want to install TensorFlow within this virtual environment complete all steps until you reach the ```pip install``` commands for tensorflow. The necessary TensorFlow dependency will automatically be installed within the virtual environment.

### Keras
(NOTE: Skip these instructions if you only want to install Keras within this virtual environment)
Keras is the high-level neural network API used to recreate the iTracker neural network. Follow the installation instructions found [here](https://keras.io/#installation). The TensorFlow backend is used for this implementation. Since Keras is supports multiple backends, the implementation should also be compatible with Theano (untested).

### Python Dependencies
The code targets (python version 3.6)[https://www.python.org/downloads/release/python-360]. 

Dependencies for this project are managed using pipenv. If you do not have pipenv installed, it can be installed using the following command.

```
pip install pipenv
```
Navigate to the iTracker directory and run the following command to create the virtual environment and install the necessary dependencies.

```
pipenv install
```
The dependencies (and their subdependencies) will be installed: ```tensorflow-gpu, keras, opencv-python```

### Machine-Specific Configuration
If there are any configuration steps that need to be executed before each runtime, create a copy of the ```config_TEMPLATE.py``` (rename to ```config.py```). Add all config code to the function ```run_config.py```. This will get executed at the start of each program execution.

If no config is required, there is no need to create the copy of the config file. The program will automatically skip this step

### Machine Learning Parameters
Create a copy of ```ml_param_TEMPLATE.json``` and rename it to ```ml_param.json```. This file contains the hyperparamters for the ML model. Configure these values before executing the code.

### Running the model
Run the command 
```
pipenv shell
```
to enter the shell of the virtual environment.
to execute the model, run the following:
```
python iTrackerKeras.py
```

