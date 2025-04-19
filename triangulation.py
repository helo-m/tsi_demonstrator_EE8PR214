import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

def triangulate(mtx1, mtx2, R, T, points1, points2, foldername):
    uvs1 = points1
    uvs2 = points2
 
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    frame1 = cv.imread(os.path.join(foldername, 'camera0_0.png'))
    frame2 = cv.imread(os.path.join(foldername, 'camera1_0.png'))
 
    # Plot the points in the images
    plt.imshow(frame1[:,:,[2,1,0]]) # BGR (opencv default) to RGB (matplotlib expected)
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
 
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()#this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    # Implements linear triangulation using the Direct Linear Transform
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
 
    p3ds = [] # 3d points are gonna be stored in here
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    print(p3ds)

    # Shift all points so the first point becomes the origin
    origin = p3ds[0]
    p3ds_shifted = p3ds - origin

    print("Shifted 3D points (first point is now origin):")
    print(p3ds_shifted)

    from mpl_toolkits.mplot3d import Axes3D

    # Plot shifted points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)

    # Axis labels
    ax.set_xlabel('X axis', color='red')
    ax.set_ylabel('Y axis', color='green')
    ax.set_zlabel('Z axis', color='blue')

    # Axis orientation arrows
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', length=2, normalize=True)   # X
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', length=2, normalize=True) # Y
    ax.quiver(0, 0, 0, 0, 0, 1, color='blue', length=2, normalize=True)  # Z

    xs = p3ds_shifted[:, 0]
    ys = p3ds_shifted[:, 1]
    zs = p3ds_shifted[:, 2]

    ax.scatter(xs, ys, zs, c='r', marker='o')

    plt.show()

    return p3ds_shifted