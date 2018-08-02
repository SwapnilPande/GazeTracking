# iTracker
Project directory for replicating iTracker Convolutional Neural Network outlined in [Eye tracking for everyone](http://gazecapture.csail.mit.edu/) (Krafka et al., 2016).

## Setup
### TensorFlow
TensorFlow is the machine learning package used to build and train the model. Follow the installation instructions found [here](https://www.tensorflow.org/install/) for your specific platform. 

NOTE: If you only want to install TensorFlow within this virtual environment complete all steps until you reach the ```pip install``` commands for tensorflow. The necessary TensorFlow dependency will automatically be installed within the virtual environment.

### Keras
NOTE: Skip these instructions if you only want to install Keras within this virtual environment

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
The dependencies (and their subdependencies) will be installed

### Machine-Specific Configuration
If there are any configuration steps that need to be executed before each runtime, create a copy of the ```config_TEMPLATE.py``` (rename to ```config.py```). Add all config code to the function ```run_config.py```. This will get executed at the start of each program execution.

If no config is required, there is no need to create the copy of the config file. The program will automatically skip this step

### Machine Learning Parameters
Create a copy of ```ml_param_TEMPLATE.json``` and rename it to ```ml_param.json```. This file contains the hyperparamters for the ML model. Configure these values before executing the code.

#### Description of parameters:
* `loadPrexistingmodel`: Flag denoted whether to train a model from beginning or to begin training from saved partially trained model
	* `prexistingModelPath`: Location of the partially trained model, only read if `loadPrexistingmodel` is `true` (can be empty string otherwise)
	* `trainLogFile`: Location of the log file associated with the trained model, only read if `loadPrexistingmodel` is `true` (can be empty string otherwise)
* `trainingHyperparameters`: Contains all of the hyperparamters to use when training - will be disregarded if `loadPrexistingmodel` is `true`
	* `learningRate`: Learning rate to use. Can either be a number, or a JSON object to schedule the learning rate to change. If passing the JSON object, the key of the object is the epoch at which to set the learning rate (0-indexed), and the value is the learning rate. NOTE: You must include a learning rate for epoch 0. 
	Example:
	```
	"learningRate" : 
		{
			"0" : 0.1,
			"10" : 0.001
		}
	```
	* `momentum` : Number for the mometum value
	* `decay` : Number for the decay value
	* `numEpochs` : Number of epochs to train the model
	* `trainBatchSize` : Batch size for training dataset
	* `validateBatchSize` : Batch size for validation dataset
	* `testBatchSize` : Batch size for test dataset
	* `trainSetProportion` : Proportion of data to use for training
	* `validationSetProportion` : Proportion of data to use for valdiation (test proportion is automtically calculated based on train and validate proportions)
* `dataPaths` : Includes data paths to various directories that are used during training
	* `pathToData` : Path to the directory containing the zipped subject directories.
	* `pathToTempDir` : Path to the directory to store the temporary unzipped dataset. If pointing to already unpacked dataset, `train.py` will search for data at `%pathToTempDir% + /temp/`
	* `pathLogging` : Path in which to store the logging data
* `machineParameters` : Includes parameters to tweak machine utilization and training performance
	* `loadTrainIntoMemory` : Boolean signifying whether or not to load the training images into memory
	* `loadValidateIntoMemory` : Boolean signifying whether or not to load the validation images into memory
	* `numberofWorkers` : Maximum number of threads to spawn on the CPU
	* `queueSize` : Maximum size of queue for batches
	* `numberofGPUS` : Number of GPUs to execute training on. NOTE: Saving model not functional for Multi-GPU execution

### Running the model
Run the command 
```
pipenv shell
```
to enter the shell of the virtual environment.
to execute the model, run the following:
```
python train.py
```

