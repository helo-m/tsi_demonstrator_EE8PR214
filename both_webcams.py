import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Disable hardware transforms for OpenCV on Windows
import cv2 as cv
import numpy as np
from scipy import linalg
import yaml

# Clear the console screen
os.system('cls' if os.name == 'nt' else 'clear')

# Global dictionary to store calibration settings
calibration_settings = {}

def parse_calibration_settings_file(filename):
    """
    Parses the calibration settings from a YAML file.

    Parameters:
        filename (str): Path to the YAML file containing calibration settings.

    Exits the program if the file does not exist or if required keys are missing.
    """
    global calibration_settings

    # Check if the file exists
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using for calibration settings:', filename)

    # Load the YAML file into the global dictionary
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # Basic validation to ensure the correct file is loaded
    if 'camera0' not in calibration_settings.keys():
        print('The key "camera0" was not found in the settings file. Verify if the correct calibration_settings.yaml file was provided.')
        quit()

def save_frames_two_cams(camera0_name, camera1_name, foldername):
    """
    Captures and saves frames from two cameras simultaneously.

    Parameters:
        camera0_name (str): Key for the first camera in the calibration settings.
        camera1_name (str): Key for the second camera in the calibration settings.
    """
    # Create the 'frames_pair' directory if it does not exist
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    # Retrieve settings for capturing frames
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']

    # Open the video streams for both cameras
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # Set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)  # Set width for camera 0
    cap0.set(4, height)  # Set height for camera 0
    cap1.set(3, width)  # Set width for camera 1
    cap1.set(4, height)  # Set height for camera 1

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        # Read frames from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras are not returning video data. Exiting...')
            quit()

        # Resize frames for display (only for visualization, not for saving)
        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            # Display instructions to ensure both cameras can see the calibration pattern
            cv.putText(frame0_small, "Ensure both cameras can see the calibration pattern", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start capturing frames", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            # Decrease the cooldown timer
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            # Save frames when cooldown reaches 0
            if cooldown <= 0:
                # Save images in PNG format for lossless compression
                savename0 = os.path.join(foldername, f'{camera0_name}_{saved_count}.png')
                savename1 = os.path.join(foldername, f'{camera1_name}_{saved_count}.png')

                # Save the frames with maximum quality
                cv.imwrite(savename0, frame0, [cv.IMWRITE_PNG_COMPRESSION, 0])  # PNG compression level 0 (no compression)
                cv.imwrite(savename1, frame1, [cv.IMWRITE_PNG_COMPRESSION, 0])  # PNG compression level 0 (no compression)

                saved_count += 1
                cooldown = cooldown_time  # Reset cooldown timer

        # Display the resized frames
        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        if k == 27:  # ESC key
            # Exit the program if ESC is pressed
            quit()

        if k == 32:  # Spacebar
            # Start capturing frames when SPACEBAR is pressed
            start = True

        # Exit the loop when the required number of frames is saved
        if saved_count == number_to_save:
            break

    # Release video streams and close all OpenCV windows
    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

def save_single_frame_two_cams(camera0_name, camera1_name, foldername):
    """
    Captures and saves one frame from two cameras simultaneously.

    Parameters:
        camera0_name (str): Key for the first camera in the calibration settings.
        camera1_name (str): Key for the second camera in the calibration settings.
    """
    # Create the 'frames_pair' directory if it does not exist
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    # Retrieve settings for capturing frames
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = 1

    # Open the video streams for both cameras
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # Set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)  # Set width for camera 0
    cap0.set(4, height)  # Set height for camera 0
    cap1.set(3, width)  # Set width for camera 1
    cap1.set(4, height)  # Set height for camera 1

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        # Read frames from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras are not returning video data. Exiting...')
            quit()

        # Resize frames for display (only for visualization, not for saving)
        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            # Display instructions to ensure both cameras can see the calibration pattern
            cv.putText(frame0_small, "Ensure both cameras have not been moved", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to capture the frame", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            savename0 = os.path.join(foldername, f'{camera0_name}_{saved_count}.png')
            savename1 = os.path.join(foldername, f'{camera1_name}_{saved_count}.png')

             # Save the frames with maximum quality
            cv.imwrite(savename0, frame0, [cv.IMWRITE_PNG_COMPRESSION, 0])  # PNG compression level 0 (no compression)
            cv.imwrite(savename1, frame1, [cv.IMWRITE_PNG_COMPRESSION, 0])  # PNG compression level 0 (no compression)
            saved_count += 1

        # Display the resized frames
        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        if k == 27:  # ESC key
            # Exit the program if ESC is pressed
            quit()

        if k == 32:  # Spacebar
            # Start capturing frames when SPACEBAR is pressed
            start = True

        # Exit the loop when the required number of frames is saved
        if saved_count == number_to_save:
            break

    # Release video streams and close all OpenCV windows
    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

def parse_settings_file():
    # Parse the calibration settings file
    parse_calibration_settings_file("calibration_settings.yaml")