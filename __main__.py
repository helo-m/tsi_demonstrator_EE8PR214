import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Disable hardware transforms for OpenCV on Windows (less delay)
from calibration import *
from both_webcams import *
from click_recognition import *
from triangulation import *

# Parses the calibration settings file
parse_settings_file()
# Captures frames from two cameras
save_frames_two_cams('camera0', 'camera1', 'stereo_frames')

images = sorted(glob.glob("stereo_frames/*"))

# Separates images for camera0 and camera1 based on filenames
images_camera0 = [img for img in images if "camera0" in os.path.basename(img)]
images_camera1 = [img for img in images if "camera1" in os.path.basename(img)]

# Calibrates each camera individually
print("Calibrating camera 0...")
mtx1, dist1, rmse1 = calibrate_camera(images_camera0)
print("Calibrating camera 1...")
mtx2, dist2, rmse2 = calibrate_camera(images_camera1)

# Performs stereo calibration
print("Performing stereo calibration...")
R, T, stereo_rmse = stereo_calibrate(mtx1, dist1, mtx2, dist2, images_camera0, images_camera1)

# Save results to a text file
with open("calibration_results.txt", "w") as f:
    f.write("===== Camera 0 Calibration =====\n")
    f.write(f"RMSE: {rmse1}\n")
    f.write(f"Camera Matrix:\n{mtx1}\n")
    f.write(f"Distortion Coefficients:\n{dist1}\n\n")

    f.write("===== Camera 1 Calibration =====\n")
    f.write(f"RMSE: {rmse2}\n")
    f.write(f"Camera Matrix:\n{mtx2}\n")
    f.write(f"Distortion Coefficients:\n{dist2}\n\n")

    f.write("===== Stereo Calibration =====\n")
    f.write(f"Stereo Calibration RMSE: {stereo_rmse}\n")
    f.write(f"Rotation Matrix (R):\n{R}\n")
    f.write(f"Translation Vector (T):\n{T}\n")

print("Calibration results saved to 'calibration_results.txt'.")

# We are gonna take one photo of the environment in each camera to select points
save_single_frame_two_cams('camera0', 'camera1', 'single_frames')

# We take the previous photos and select points in them
# IMPORTANT: The points should be chosen in the same order for both
# Press ESC to exit after choosing the points
points0 = click_recognize('.\single_frames\camera0_0.png')
points1 = click_recognize('.\single_frames\camera1_0.png')

# Makes the triangulation of the chosen points based on 'real life' arbitrary 3d coordinates
# PS: The first points chosen will serve as origin
points3d = triangulate(mtx1, mtx2, R, T, points0, points1, '.\single_frames')