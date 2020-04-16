import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#images_all = ["2.jpg","3.jpg","4.jpg","5.jpg"]
#for z in images_all:
#    img = cv2.imread(z)

def obj_detector(image):
        response, image = capture.read()	
        image = cv2.resize(image, None, fx=0.95, fy=0.95)
        height, width, channels = image.shape

# Detecting objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

# Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

            # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 1, color, 3)
                #image = cv2.flip(image, 1)		
                return image

capture = cv2.VideoCapture(0)  
cv2.waitKey(0)  
    
while True:
        response, frame = capture.read()
        cv2.imshow("Live object detection", obj_detector(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()
    
    

    
    
