import sys, pygame
import os

if os.name == 'nt':
	import ctypes
	ctypes.windll.user32.SetProcessDPIAware()


class iTrackerUI:

	#Creates the nececssary pygame UI elements to draw the UI for iTracker
	def __init__(self):
		#Initializing pygame UI
		pygame.init()
		
		self.dispInfo = pygame.display.Info() #Collecting info about the display application is running on
		self.screenSize = self.wScreenPx, self.hScreenPx = (self.dispInfo.current_w, self.dispInfo.current_h) #Storing screen size from info object
		self.black = 0,0,0 #Black tuple for writing background color

		#Load image for dots and cursor and scale to 100x100 pixels
		self.dotSize = (100,100)
		self.cursorSize = (100,100)
		self.dot = pygame.transform.scale(pygame.image.load("UI_assets/pokeball.png"), self.dotSize) 
		self.cursor = pygame.transform.scale(pygame.image.load("UI_assets/greatball.png"), self.cursorSize) 	

		#Generating coordinates for dots on screen relative to screen size
		self.x1 = int(self.wScreenPx*0.25 - self.dotSize[0]/2) 
		self.x2 = int(self.wScreenPx*0.75 - self.dotSize[0]/2)
		self.y1 = int(self.hScreenPx*0.25 - self.dotSize[1]/2)
		self.y2 = int(self.hScreenPx*0.75 - self.dotSize[1]/2)
		self.dotCoords = [(self.x1, self.y1), (self.x2, self.y1), (self.x1, self.y2), (self.x2, self.y2)]
		
		#Generating initial coordinates for the cursor
		self.cursorX = int(self.wScreenPx*0.5 - self.cursorSize[0]/2) 
		self.cursorY = int(self.hScreenPx*0.5 - self.cursorSize[1]/2)
		self.cursorCoords = (self.cursorX, self.cursorY) 

		#Create screen surface object to display
		self.screen = pygame.display.set_mode(self.screenSize, pygame.FULLSCREEN)

		#Define camera and screen parameters
		self.xCameraOffsetCm = 14.25 #Horizonatl displacement of camera relative to origin pixel in cm
		self.yCameraOffsetCm = -0.75 #Vertical displacement of camera relative to origin pixel in centimeters
		self.wScreenCm = 28.5 #Width of the screen in cm
		self.hScreenCm = 19.25 #Height of the screen in cm

		#Conversaion factors to scale centimeters to screen pixels
		self.xCm2Px = self.wScreenPx/self.wScreenCm
		self.yCm2Px = self.hScreenPx/self.hScreenCm

		self.clock = pygame.time.Clock() #Initialize clock for calculating fps
		self.font = pygame.font.Font(None, 60) #Create font for writing fps on screen


	# cm2Px
	# Transforms x and y coordinates in centimeters (relative to camera) to pixels (relative to origin pixel)
	# Arugments:
	# 	coods - tuple containing coordinates in centimeters relative to camera
	# Returns tuple containing transformed coordinates relative to origin pixel (and graphics coordinate system)
	def cm2Px(self, coords):
		return (round(self.xCm2Px*(coords[0] + self.xCameraOffsetCm) - self.cursorSize[0]/2),
				round(self.yCm2Px*(-1*coords[1] + self.yCameraOffsetCm) - self.cursorSize[1]/2))

	# updateCursor
	# Redraws the UI with the updated cursor location
	# updatedCursorCoords - tuple containing x and y coordinates of the cursor
	# If no coordinates are passed, display is updated with previous cursor coordinates
	def updateCursor(self, updatedCursorCoords = None):
		#Looks for exit event to close application
		for event in pygame.event.get():
			if event.type == pygame.QUIT: sys.exit()

		#Clears the screen to draw update
		self.screen.fill(self.black)

		#Draw dots on screen
		for dotCoord in self.dotCoords:
			self.screen.blit(self.dot, dotCoord)
			
		#Store updated cursor coordinates
		if(updatedCursorCoords):
			self.cursorCoords = self.cm2Px(updatedCursorCoords)
		
		#Draw cursor to screen
		self.screen.blit(self.cursor, self.cursorCoords)

		#Measuring FPS
		self.clock.tick(60) #Tick clock timer
		fps = self.font.render(str(int(self.clock.get_fps())), True, pygame.Color('white')) #Get fps
		self.screen.blit(fps, (0, 0)) #Draw FPS on screen

		#Update screen
		pygame.display.flip()


ui = iTrackerUI();
while 1:
	for x,y in zip(range(1000, 3000), range(1000, 3000)):
		ui.updateCursor((14.25, -20))




