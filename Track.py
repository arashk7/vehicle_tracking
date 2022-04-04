import cv2
import numpy as np

''' Load Homography matrix from file'''
H = np.load('H4.npz')['arr_0']
print(H)

''' Load sample BEV image '''
ge_img_orig = cv2.imread('google-earth.png')

''' Load the output of DeepSORT tracker '''
label_file_name = 'video_01.txt'
lines = []
vehicles = {}
frame_items = []
prev_frame_id = 2
with open('DeepSORT_result/' + label_file_name) as f:
    lines = f.readlines()
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
    frame_items.append({'id': id, 'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h})

    prev_frame_id = frame_id

'''Load the video file'''
cap = cv2.VideoCapture('video_01.mp4')

if not cap.isOpened():
    print("Error opening video file")

frame_counter = 0

frame_id = 1

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
                cv2.circle(ge_img, (px1, py1), 10, colors[class_id], 3)
                cv2.putText(ge_img, classes[class_id] + ' ' + str(id), (px1, py1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

                ''' Drawing vehicle on warped view '''
                cv2.rectangle(cam_frame, (x, y), (int(x + (w)), int(y + (h))), colors[class_id], 2)
                cv2.putText(cam_frame, classes[class_id] + ' ' + str(id), (x, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

                ''' Drawing vehicle on Perspective view '''
                cv2.circle(warped, (px1, py1), 10, colors[class_id], 3)
                cv2.putText(warped, classes[class_id] + ' ' + str(id), (px1, py1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.8, thickness=2)

        ''' Resize and show the images '''
        wnd_size = (int(width / 1.5), int(height / 1.5))

        ''' Display the perspective view frame '''
        cam_frame = cv2.resize(cam_frame, wnd_size)
        cv2.imshow('cam view', cam_frame)

        ''' Display the Google Earth view frame '''
        ge_img = cv2.resize(ge_img, wnd_size)
        cv2.imshow('ge view', ge_img)

        ''' Display the warped view frame '''
        warped = cv2.resize(warped, wnd_size)
        cv2.imshow('warped', warped)

        frame_counter += 1
        frame_id += 1

    ''' Press Q on keyboard to  exit '''
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

''' When everything done, release the video capture object'''
cap.release()

'''Closes all the frames'''
cv2.destroyAllWindows()
