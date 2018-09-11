import argparse #Argument parsing

#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument('image_directory', help = 'directory in which to save captured images')
parser.add_argument('-d', '--number-of-dots', help = 'number of dots to iterate over', default = '5')
parser.add_argument('-t', '--time-per-dot', help = 'time to display each dot', default = '10')
args = parser.parse_args()

from utils.uiUtils import dataCollectionUI #Importing data collection UI object

ui = dataCollectionUI(args.image_directory, float(args.time_per_dot), 0.5)

images = 0
while images < int(args.number_of_dots):
    images = ui.updateUI()
ui.writeData()