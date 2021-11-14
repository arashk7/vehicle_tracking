import cv2
import numpy as np

''' Camera Keypoints '''
cam_pts = []

''' Google earth Keypoints'''
ge_pts = []


def mouse_event_cam(event, x, y, flags, param):
    ''' Mouse event on camera window'''
    if event == cv2.EVENT_LBUTTONDOWN:
        cam_pts.append([x, y])
        print('cam_click')


def mouse_event_ge(event, x, y, flags, param):
    ''' Mouse event on BEV window'''
    if event == cv2.EVENT_LBUTTONDOWN:
        ge_pts.append([x, y])
        print('ge_click')


''' Load the images from files'''
ge_img = cv2.imread('google-earth.png')
cam_img = cv2.imread('camera-view.png')

''' Get the Maximum width and height of the images '''
maxWidth = max(cam_img.shape[1], ge_img.shape[1])
maxHeight = max(cam_img.shape[0], ge_img.shape[0])

''' Define the window names'''
cv2.namedWindow("cam_view")
cv2.namedWindow("ge_view")

''' Set the mouse event functions '''
cv2.setMouseCallback("cam_view", mouse_event_cam)
cv2.setMouseCallback("ge_view", mouse_event_ge)

''' Show images '''
cv2.imshow('cam_view', cam_img)
cv2.imshow('ge_view', ge_img)

src_pts = np.array(cam_pts)
dst_pts = np.array(ge_pts)

''' Uncomment two line below to calculate the homography'''
# H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# np.savez('H4.npz',H)

''' Load the homograhy from file '''
H = np.load('H4.npz')['arr_0']
print(H)

''' Warp the perspective view '''
warped = cv2.warpPerspective(cam_img, H, (maxWidth, maxHeight))

cv2.imshow('result', warped)

cv2.waitKey(0)
