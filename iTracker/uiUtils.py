#uiUtils contains definitions for user interface utilities 
#Abstracts these functions so that it is not needed to import libraries in each file

from progressbar import ProgressBar 

# yesNoPrompt
# Accepts a yes or no (y/n) input from the user and returns a boolean with result
def yesNoPrompt():
	response = input()
	while(response.lower() != 'y' and response.lower() != 'n'):
		print("Enter only y or n:")
		response = input()
	return (response == 'y')

# listOptionsPrompt
# Accepts a number input to select an option from a list
# 
def listOptionsPrompt(options):
	for option in options:

	response = input()
	while(response.lower() != 'y' and response.lower() != 'n'):
		print("Enter only y or n:")
		response = input()
	return (response == 'y')

# createProgress Bar
# Initializes a progress bar object with an optional max value
# maxVal - Optional max value for progress bar
# 			If no maxVal is passed, progressBar will be initialized without max
# Returns ProgressBar Object
def createProgressBar(maxVal = None):
	if(maxVal != None):
		return ProgressBar(maxval = maxVal)
	return ProgressBar()