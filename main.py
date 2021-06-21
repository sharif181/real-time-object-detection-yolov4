import cv2
import numpy as np

print("Loading YOLO")
net = cv2.dnn.readNet("yolov4.weights","yolov4.cfg")
classes = []
with open("coco.names","r") as f:
  classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers() ]
print("YOLO loaded") 


video = cv2.VideoCapture(0)
video.set(3,600)
video.set(4,720)
video.set(10,100)


while True:
  re,img = video.read()
  height,width,channels = img.shape
  blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)

  classids = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
      scores = detection[5:]
      classid = np.argmax(scores)
      confidence = scores[classid]
      if confidence > 0.5:
        center_x = int(detection[0]*width)
        center_y = int(detection[1]*height)
        w = int(detection[2]*width)
        h = int(detection[3]*height)
        x = int(center_x-w/2)
        y = int(center_y-h/2)
        boxes.append([x,y,w,h])
        confidences.append(float(confidence))
        classids.append(classid)
  indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
  colors = np.random.uniform(0,255,size=(len(classes),3))
  for i in range(len(boxes)):
    if i in indexes:
      x,y,w,h = boxes[i]
      label = str(classes[classids[i]])
      color = colors[classids[i]]
      cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
      cv2.putText(img,label+" "+str(round(confidences[i]*100,2))+"%",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1/2,color,2)
  cv2.imshow("Output",img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()