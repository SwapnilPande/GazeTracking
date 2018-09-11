import argparse #Argument parsing

#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument('image_directory', help = 'directory in which to save captured images')
args = parser.parse_args()

from utils.uiUtils import dataCollectionUI #Importing data collection UI object

ui = dataCollectionUI(args.image_directory, 1, 0.5)

images = 0
while images < 2:
    images = ui.updateUI()
ui.writeData()