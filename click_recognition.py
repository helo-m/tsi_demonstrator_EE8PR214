import cv2 as cv

coordX = []
coordY = []

# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, img): 
	# checking for left mouse clicks 
	if event == cv.EVENT_LBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y)
		coordX.append(x)
		coordY.append(y)

		# displaying the coordinates 
		# on the image window 
		font = cv.FONT_HERSHEY_SIMPLEX 
		cv.putText(img, str(x) + ',' +
					str(y), (x,y), font, 
					1, (255, 0, 0), 2) 
		cv.imshow('image', img) 

def click_recognize(img_path):
	img = cv.imread(img_path, 1)
	cv.namedWindow("image", cv.WINDOW_NORMAL)
	while(1):
		# displaying the image 
		cv.imshow('image', img) 

		# setting mouse handler for the image 
		# and calling the click_event() function 
		cv.setMouseCallback('image', click_event, img) 

		# wait for a key to be pressed to exit 
		if cv.waitKey(0):
			break

		# close the window 
		cv.destroyAllWindows()

	points_tuples = list(map(lambda x, y: [x, y], coordX, coordY))

	print(points_tuples)

	coordX.clear()
	coordY.clear()

	return points_tuples