import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_doc(path):
    doc = open(path , 'r')
    txt = doc.read()
    doc.close()
    return txt

def find_masks(yolo_model , img , colors , class_labels , img_height , img_width):
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
    output_img = img
    for max_value_id in max_value_ids:
        max_class_id = max_value_id[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
        prediction_confidence = prediction_confidence*100
        text = predicted_class_label + ":" + str(int(prediction_confidence)) + "%"
        color_box = colors[predicted_class_id]
        color_box = (int(color_box[0]), int(color_box[1]), int(color_box[2])) 
        output_img = cv2.rectangle(output_img , (start_x_pt , start_y_pt) , (start_x_pt+box_width , start_y_pt+box_height) , color_box , thickness = 4)
        output_img = cv2.putText(output_img , text , (start_x_pt , start_y_pt-4) , cv2.FONT_HERSHEY_SIMPLEX , 1 , color = color_box , thickness = 2)
    return output_img

cfg = 'mask_yolov4.cfg'
weight = 'mask_yolov4_best.weights'
class_labels = ['without_mask' , 'with_mask']
colors = [[0 , 0 , 255] , [0 , 255 , 0]]
yolo_model = cv2.dnn_DetectionModel(cfg , weight)
cap = cv2.VideoCapture(0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap = cv2.VideoCapture(0)
print("Hello")
if cap.isOpened()== False:
	print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
#writer = cv2.VideoWriter(r"output.mp4", cv2.VideoWriter_fourcc(*'XVID'),2, (width, height))
while cap.isOpened():
	res , frame = cap.read()
	if res == False:
		break
	out_frame = find_masks(yolo_model , frame , colors , class_labels , height , width)
	#writer.write(out_frame)
	cv2.imshow('frame' , out_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
#writer.release()
cv2.destroyAllWindows()
