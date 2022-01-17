import copy

import cv2
import numpy as np
import math
from ItemAnalysis import ItemAnalysis


class Axis3D:
    def __init__(self):
        print('3D Axis')

    def Unproject(self, points, Z, intrinsic, distortion):
        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]

        ''' Step 1. Undistort. '''
        points_undistorted = np.array([])
        if len(points) > 0:
            points_undistorted = cv2.undistortPoints(np.expand_dims(np.float32(points), axis=1), intrinsic, distortion,
                                                     P=intrinsic)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        ''' Step 2. Reproject.'''
        result = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[0] if len(Z) == 1 else Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            result.append([x, y, z])
        return result

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy, oz = origin
        px, py, pz = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        qz = pz
        return qx, qy, qz

    def draw_BB(self, img, rvec, pt1, pt2, _Z, ang):
        pt1 = pt1 - [0, 0]
        pt2 = pt2 + [0, 0]
        ''' Size of the Boounding Box '''
        sizex = pt2[0] - pt1[0]
        sizey = pt2[1] - pt1[1]

        Z = [_Z]

        ''' Unproject the center point to 3D with sample Z'''
        point_3d = self.Unproject(pt1,
                                  Z,
                                  mtx, dist)

        ''' Set the translation vector '''
        tvec = np.array([[point_3d[0][0], point_3d[0][1], point_3d[0][2]]])

        ''' Set the BB size '''
        x1 = 0
        y1 = 0
        x2 = 4
        y2 = 2

        ''' Bounding Box points '''
        BB_base = np.float32([[x1, y1, 1], [x1, y2, 1], [x2, y2, 1], [x2, y1, 1],
                              [x1, y1, 0], [x1, y2, 0], [x2, y2, 0], [x2, y1, 0]])

        bb_base_center = np.average(BB_base, axis=0)

        for i in range(8):
            BB_base[i] = self.rotate(np.array([bb_base_center[0], bb_base_center[1], BB_base[i][2]]), BB_base[i], ang)

        ''' Project the new translation point to the 2D view '''
        imgpts, jac = cv2.projectPoints(BB_base, rvec, tvec, mtx, dist)

        imgpts = np.float32(imgpts).reshape(-1, 2)

        ''' Set the 3D bounding Box based on 2D Bounding Box '''
        x = pt1[0]
        y = pt1[1]

        minx = min(imgpts[:, 0])
        miny = min(imgpts[:, 1])
        maxx = max(imgpts[:, 0])
        maxy = max(imgpts[:, 1])

        distx = maxx - minx
        disty = maxy - miny

        scalex = sizex / distx
        scaley = sizey / disty

        imgpts[:, 0] = np.float32(imgpts[:, 0] * scalex)
        imgpts[:, 1] = np.float32(imgpts[:, 1] * scaley)

        minx = min(imgpts[:, 0])
        miny = min(imgpts[:, 1])

        stdistx = x - minx
        stdisty = y - miny

        imgpts[:, 0] += stdistx
        imgpts[:, 1] += stdisty

        imgpts_botton = np.int32([imgpts[:4]])
        imgpts_top = np.int32([imgpts[4:]])

        ''' Draw ground floor in green'''
        img = cv2.drawContours(img, imgpts_botton, -1, (0, 255, 0), 3)

        ''' Draw pillars in blue color'''
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(np.int32(imgpts[i])), tuple(np.int32(imgpts[j])), (255), 3)

        ''' Draw top layer in red color'''
        img = cv2.drawContours(img, imgpts_top, -1, (0, 0, 255), 3)
        return img


''' Load Homography matrix from file'''
H = np.load('H4.npz')['arr_0']
print(H)

''' Correspondence feature points '''
cam_pts = np.load('CAM_PTS.npz')['arr_0']
ge_pts = np.load('GE_PTS.npz')['arr_0']

''' Camera Calibration '''
ge_points = []  # 3d point in real world space
cam_points = []  # 2d points in image plane.

''' Prepare the set of points for calibration '''
for pts in ge_pts:
    ge_points.append([pts[0], pts[1], 0])
for pts in cam_pts:
    cam_points.append([[pts[0], pts[1]]])
ge_points = np.array([ge_points], dtype=np.float32)
cam_points = [np.array(cam_points, dtype=np.float32)]

''' Sample image used for calibration (used get the size) '''
cam_img = cv2.imread('camera-view.png')

''' Camera calibration '''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(ge_points, cam_points, cam_img.shape[::-1][1:], None, None)

''' Load sample BEV image '''
ge_img_orig = cv2.imread('google-earth.png')

''' Load the output of DeepSORT tracker '''
label_file_name = 'video_01.txt'
lines = []
vehicles = {}
frame_items = []
prev_frame_id = 2

'''Open the DeepSORT result file'''
with open('DeepSORT_result/' + label_file_name) as f:
    lines = f.readlines()

''' Reading the DeepSORT result file line by line'''
for line in lines:
    splite_line = line.split(' ')
    frame_id = int(splite_line[0])

    id = int(splite_line[1])
    x = int(splite_line[2])
    y = int(splite_line[3])
    w = int(splite_line[4])
    h = int(splite_line[5])
    class_id = int(splite_line[6])

    if not frame_id == prev_frame_id:
        vehicles[frame_id - 1] = frame_items
        frame_items = []
    frame_items.append({'id': id, 'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h, 'ang': 0.0})

    prev_frame_id = frame_id

'''Load the video file'''
cap = cv2.VideoCapture('video_01.mp4')

if not cap.isOpened():
    print("Error opening video file")

frame_counter = 0

frame_id = 1

''' Initialize 3D bounding box drawer '''
axis3d = Axis3D()

''' Analysis any change on the items positions '''
item_analysis = ItemAnalysis()

''' Video Loop '''
while cap.isOpened():
    ''' make copy of the Google Earth view '''
    ge_img = ge_img_orig.copy()
    ''' Capture frame from video '''
    ret, cam_frame = cap.read()

    if ret:
        ''' Get the size of the image '''
        width = cam_frame.shape[1]
        height = cam_frame.shape[0]

        ''' Warp the perspective view using Homography matrix '''
        warped = cv2.warpPerspective(cam_frame, H, (width, height))

        ''' Check frame ID limits'''
        if frame_id < 2:
            frame_id += 1
            continue
        if frame_id > 1248:
            frame_id += 1
            break
        ''' Get a list of the current frame's vehicles'''
        vehicle_items = vehicles[frame_id]

        ''' Looping on vehicles '''
        for item in vehicle_items:
            id = item['id']
            class_id = item['class_id']
            x = item['x']
            y = item['y']
            w = item['w']
            h = item['h']
            ang = item['ang']
            cx = x + (w / 2)
            cy = y + (h / 2)

            ''' Transform the position of the vehicle '''
            p1 = cv2.perspectiveTransform(np.array([[[x + (w / 2), y + (h / 2)]]], dtype=float), H).ravel()
            px1 = int(p1[0])
            py1 = int(p1[1])

            ''' Select between different vehicle type '''
            classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

            ''' Coloring different vehicle with different color '''
            colors = {2: (0, 255, 255), 3: (255, 0, 255), 5: (0, 0, 255), 7: (255, 0, 0)}

            if class_id in classes.keys():  # (car, motorcycle, bus, truck)
                ''' Drawing vehicle on Google Earth view '''
                item_analysis.add_item(id, px1, py1)
                rot = item_analysis.get_rot_dist(id)

                # print(rot)
                cv2.circle(ge_img, (px1, py1), 10, colors[class_id], 3)
                cv2.putText(ge_img, str(rot[0]), (px1, py1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

                ''' Drawing vehicle on Perspective view '''
                # cv2.rectangle(cam_frame, (x, y), (int(x + (w)), int(y + (h))), colors[class_id], 2)
                axis3d.draw_BB(cam_frame, rvecs[0], np.array([x, y]), np.array([x + w, y + h]), 30, rot[0])
                cv2.putText(cam_frame, classes[class_id] + ' ' + str(id), (x, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

                ''' Drawing vehicle on warped view '''
                cv2.circle(warped, (px1, py1), 10, colors[class_id], 3)
                cv2.putText(warped, classes[class_id] + ' ' + str(id), (px1, py1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

        ''' Update items' list every 10 frames '''
        if frame_id % 10 == 0:
            item_analysis.step()

        ''' Resize and show the images '''
        wnd_size = (int(width / 2), int(height / 2))

        ''' Display the perspective view frame '''
        cam_frame = cv2.resize(cam_frame, wnd_size)
        cv2.imshow('cam_view', cam_frame)

        ''' Display the Google Earth view frame '''
        ge_img = cv2.resize(ge_img, wnd_size)
        cv2.imshow('ge_view', ge_img)

        ''' Display the warped view frame '''
        warped = cv2.resize(warped, wnd_size)
        cv2.imshow('warped', warped)

        frame_counter += 1
        frame_id += 1

    ''' Press Q on keyboard to  exit '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
''' When everything done, release the video capture object'''
cap.release()

'''Closes all the frames'''
cv2.destroyAllWindows()
