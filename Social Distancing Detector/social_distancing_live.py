
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import math
import time
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3
HEIGHT_ROI = 600
WIDTH_ROI = 700




def load_doc(path):
    doc = open(path , 'r')
    txt = doc.read()
    doc.close()
    return txt




def four_point_transform(pts , maxHeight , maxWidth):
    (tl, tr, br, bl) = pts
    pts = np.float32(pts)
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return M



def compute_point_perspective_transformation(matrix,list_downoids):
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return transformed_points_list




def draw_rectangle(corner_points , frame):
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
    cv2.line(frame, (corner_points[2][0], corner_points[2][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[0][0], corner_points[0][1]), COLOR_BLUE, thickness=1)




def selectROI(VideoPath):
    pts = []
    def CallBackFunc(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
            pts.append([x,y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
            pts.append([x,y])

    windowName = 'MouseCallback'
    cv2.namedWindow(windowName)

    cv2.setMouseCallback('MouseCallback' , CallBackFunc)

    cap = cv2.VideoCapture(VideoPath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if cap.isOpened()== False: 
        print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
    while cap.isOpened():
        res , frame = cap.read()
        if res == False:
            break
        new_frame = frame
        for p in pts:
            p = tuple(p)
            new_frame = cv2.circle(new_frame , p , 8 , (255 , 0 , 0) , thickness = -1)
        cv2.imshow('MouseCallback' , new_frame)
        time.sleep(1/20)
        if len(pts) == 4:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pts





def find_objects_yolo(yolo_model , img , class_labels , img_height , img_width):
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layers = yolo_model.getUnconnectedOutLayersNames()
    blob_img = cv2.dnn.blobFromImage(img , scalefactor = 1/255 , size = (416 , 416) , swapRB = True , crop = False)
    yolo_model.setInput(blob_img)
    obj_detection_layers = yolo_model.forward(yolo_output_layers)
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    for obj_det_layer in obj_detection_layers:
        for obj in obj_det_layer:
            scores = obj[5:]
            bounding_box = obj[0:4] * np.array([img_width , img_height , img_width , img_height])
            (box_centre_x_pt , box_centre_y_pt , box_width , box_height) = bounding_box.astype("int")
            start_x_pt = int(box_centre_x_pt - (box_width/2))
            start_y_pt = int(box_centre_y_pt - (box_height/2))
            predicted_class_id = np.argmax(scores)
            confidence_score = float(scores[predicted_class_id])
            if confidence_score > 0.40:
                class_ids_list.append(predicted_class_id)
                confidences_list.append(confidence_score)
                boxes_list.append([start_x_pt , start_y_pt , int(box_width) , int(box_height)])
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list , confidences_list , 0.5 , 0.4)
    output = []
    for max_value_id in max_value_ids:
        max_class_id = max_value_id[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        if predicted_class_label != 'person':
            continue
        prediction_confidence = confidences_list[max_class_id]
        prediction_confidence = prediction_confidence*100
        output.append([predicted_class_label , prediction_confidence , box])
    return output





def get_points_from_box(box):
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]
    center_x = int(start_x_pt + (box_width/2))
    center_y = int(start_y_pt + (box_height/2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = int(center_y - (box_height/2))
    return (center_x,center_y),(center_x,int(center_y_ground))





def get_centroids_and_groundpoints(array_boxes_detected):
    array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
    for index,box in enumerate(array_boxes_detected):
        centroid,ground_point = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids,array_groundpoints





def check_social_distancing(VideoPath , distance_minimum = 90):
    cfg_org = 'yolov4.cfg'
    weight_org = 'yolov4.weights'
    yolo_model_org = cv2.dnn.readNetFromDarknet(cfg_org , weight_org)
    yolo_model_org.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    yolo_model_org.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    txt = load_doc('coco_classes.txt')
    class_labels_org = txt.split('\n')
    colors = [[0 , 0 , 255] , [0 , 255 , 0]]
    pts = selectROI(VideoPath)
    matrix = four_point_transform(pts , HEIGHT_ROI , WIDTH_ROI)
    cap = cv2.VideoCapture(VideoPath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = HEIGHT_ROI
    #width = WIDTH_ROI
    if cap.isOpened()== False: 
        print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
    #writer1 = cv2.VideoWriter(r"C:\Users\goeld\Computer-Vision-with-Python\mask_detection\output_social.mp4", cv2.VideoWriter_fourcc(*'XVID'),15, (width, height))
    #writer2 = cv2.VideoWriter(r"C:\Users\goeld\Computer-Vision-with-Python\mask_detection\output_bird.mp4", cv2.VideoWriter_fourcc(*'XVID'),15, (WIDTH_ROI, HEIGHT_ROI))
    while cap.isOpened():
        res , frame = cap.read()
        #frame = cv2.resize(frame , (height , width))
        if res == False:
            break
        output_org = find_objects_yolo(yolo_model_org , frame , class_labels_org , height , width)
        array_of_boxes = []
        for i in output_org:
            array_of_boxes.append(i[2])
        array_centroids,array_groundpoints = get_centroids_and_groundpoints(array_of_boxes)
        transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
        out_screen = np.zeros((HEIGHT_ROI , WIDTH_ROI , 3) , dtype = np.float32)
        for point in transformed_downoids:
            x,y = point
            cv2.circle(out_screen, (x,y), BIG_CIRCLE, COLOR_GREEN, 2)
            cv2.circle(out_screen, (x,y), SMALL_CIRCLE, COLOR_GREEN, -1)
        output_img = frame
        for index,downoid in enumerate(transformed_downoids):
                if not (downoid[0] > WIDTH_ROI or downoid[0] < 0 or downoid[1] > HEIGHT_ROI or downoid[1] < 0 ):
                    p1 = (array_of_boxes[index][0] , array_of_boxes[index][1])
                    p2 = (array_of_boxes[index][0] + array_of_boxes[index][2] , array_of_boxes[index][1] + array_of_boxes[index][3])
                    cv2.rectangle(output_img,p1,p2,COLOR_GREEN,2)
        #output_img = draw_boxes(output_img , output_org , colors)
        list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
        for i,pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
            # Check if the distance between each combination of points is less than the minimum distance chosen
            if math.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
                # Change the colors of the points that are too close from each other to red
                if not (pair[0][0] > WIDTH_ROI or pair[0][0] < 0 or pair[0][1] > HEIGHT_ROI  or pair[0][1] < 0 or pair[1][0] > WIDTH_ROI or pair[1][0] < 0 or pair[1][1] > HEIGHT_ROI  or pair[1][1] < 0):
                    x = pair[0][0]
                    y = pair[0][1]
                    cv2.circle(out_screen, (x,y), BIG_CIRCLE, COLOR_RED, 2)
                    cv2.circle(out_screen, (x,y), SMALL_CIRCLE, COLOR_RED, -1)
                    x = pair[1][0]
                    y = pair[1][1]
                    cv2.circle(out_screen, (x,y), BIG_CIRCLE, COLOR_RED, 2)
                    cv2.circle(out_screen, (x,y), SMALL_CIRCLE, COLOR_RED, -1)
                    # Get the equivalent indexes of these points in the original frame and change the color to red
                    index_pt1 = list_indexes[i][0]
                    index_pt2 = list_indexes[i][1]
                    p1 = (array_of_boxes[index_pt1][0] , array_of_boxes[index_pt1][1])
                    p2 = (array_of_boxes[index_pt1][0] + array_of_boxes[index_pt1][2] , array_of_boxes[index_pt1][1] + array_of_boxes[index_pt1][3])
                    cv2.rectangle(output_img,p1,p2,COLOR_RED,2)
                    p1 = (array_of_boxes[index_pt2][0] , array_of_boxes[index_pt2][1])
                    p2 = (array_of_boxes[index_pt2][0] + array_of_boxes[index_pt2][2] , array_of_boxes[index_pt2][1] + array_of_boxes[index_pt2][3])
                    cv2.rectangle(output_img,p1,p2,COLOR_RED,2)
        draw_rectangle(pts , output_img)
        #writer1.write(output_img)
        cv2.imshow('frame' , output_img)
        #writer2.write(out_screen)
        cv2.imshow('frame2' , out_screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #writer1.release()
    #writer2.release()
    cv2.destroyAllWindows()





check_social_distancing(r"video.avi" , 100)













