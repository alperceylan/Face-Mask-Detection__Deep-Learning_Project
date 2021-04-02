
import cv2
import numpy as np


#   To make your operation with WebCam, into "v2.Video Capture ()"; We write
# "0" instead of the video address. With "0" we reach the webcam:
md = cv2.VideoCapture(0)
#   If we want to work on a video, we need to write the address of that video
# in brackets.


while True:
    #   Here, "md.read ()" returns two values. The first is "True / False".
    # If the frames are read correctly, "frame true" returns true and synchronizes
    # the variable to "frame". If it was read incorrectly, "ret false" returns
    # false and the "frame" cannot be read:
    ret, frame = md.read()

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    #   We also need to convert each frame to "blob" format:
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    #   We will write the names of the tags (objects) that the developed model
    # can recognize:
    labels = ["Mask", "Without Mask", "without Mask", "Mask"]

    #   Next, create a different color for each label:
    colors = ["0,128,0", "0,0,250", "0,0,250", "0,128,0"]
    #   We need to roam each element in the "colors" variable one by one and
    # turn it into an "integer". Because each element above is a "string"
    # expression. And first we need to separate these numbers from the "commas"
    # between them:
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    #   Now I want all these elements to be in a single array:
    colors = np.array(colors)
  

    #   To start using the addresses of the Python files named "cfg" and "weight"
    # that we downloaded earlier, by typing them into the function:
    model = cv2.dnn.readNetFromDarknet("C:\YOLO_Mask_Detection\yolov3_mask.cfg",
                                       "C:\YOLO_Mask_Detection\yolov3_mask_last.weights")
    
    layers = model.getLayerNames()
    #   Right now, all layers are available in this "layers" variable that we
    # assign. However, we first want to use layers that we can "detect". These
    # layers are also "OUT" layers:
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    #   With the above "model.getUnconnectedOutLayers ()" method, we found our
    # "OUT" layers. To find OUT layers, by using "-1" operation; we wanted the
    # result found to come one step back. The reason for this is that while the
    # program was finding the OUT layers, it counted the elements in the "layers"
    # list starting from "1" and gave us its place in that way. However, in "index"
    # operations, the first element is in "0. index". For this, we wanted to directly
    # get the OUT layers we wanted by adding "-1" to the equation.
    # (We can check it in "Variable explorer" section.)
    
    #   To give the "Blob", which we previously converted into 4-Dimensional
    # Tensor, into our model:
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)
    #   What we do in this line; Reaching some values in OUT LAYERS! And it is
    # to assign these values into the "detection_layers" variable we created.


    #============= NON-MAXIMUM SUPPRESSION -- OPERATION 1 ====================
    #   *NOTE*  ==  predicted_ID = ids
    ids_list = []
    boxes_list = []
    confidences_list = []
    #======================== END OF OPERATION 1 =============================


    #   With the following "for" loop, we will traverse the arrays inside
    # "detection_layers" one by one:
    for detection_layer in detection_layers:
        #   In this "for" loop, we will also traverse the values in these
        # array sone by one:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            #   Since the first 5 values are "BOUNDING BOX" values, we want to
            # skip those values and keep the next SCORE values. This is all about
            # the mathematical background of the YOLO algorithm.
            
            predicted_id = np.argmax(scores)
            #   We use this method to find the index of the maximum valued
            # argument held by the "scores" variable. This maximum value will
            # be our predicted value!
            
            confidence = scores[predicted_id]
            #   We perform this assignment to access the trust score.
            
            if confidence > 0.20:
                
                lbl = labels[predicted_id]
                
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width,
                                                                 frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                #   The first 4 values are our "bounding box" values. In order
                # to have meaningful values of the "Bounding Box", we need to
                # multiply our first 4 arguments by our width and height arguments.
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                
                #============= NON-MAXIMUM SUPPRESSION -- OPERATION 2 ====================
                #   We will fill in the empty lists we opened in "operation 1"
                # above:
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                #======================== END OF OPERATION 2 =============================
            
            
            
    #============= NON-MAXIMUM SUPPRESSION -- OPERATION 3 ====================
    #   ****NOTE****
    #   The function "cv2.dnn.NMSBoxes()" below returns the IDs of the highest
    # reliability rectangles. That is, it returns Bounding Boxes with MAXIMUM
    # CONFIDENCE as an array:
    # cv2.dnn.NMSBoxes(list of bounding boxes, list of trust lists, trust score, Threshold) -->
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    #   Currently, there are IDs of rectangles with the highest confidence score
    # in "max_ids"!
     
    for max_id in max_ids:
        
        #   We access a value in "max_ids" with index: 
        max_class_id = max_id[0]
        
        #   From the "boxes_list" we will access the "values of the box belonging
        # to the object":
        box = boxes_list[max_class_id]
        #   This "box" variable is used for the Bounding Boxes belonging to the
        # object; It will be keeping the values such as starting point, end point,
        # width and length in order:
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
        
        #   We will access the values predicted from the "ids_list" one by one:
        predicted_id = ids_list[max_class_id]
        
        #   We have accessed the lbl that we "detect" from within "labels"
        # with the index value it holds in the "predicted_id" variable:
        lbl = labels[predicted_id]
        
        #   Using the variable "max_class_id", we accessed the "confidence"
        # value of our corresponding "lbl" from within "confidences_list":
        confidence = confidences_list[max_class_id]
  
    #======================== END OF OPERATION 1 =============================


        end_x = start_x + box_width
        end_y = start_y + box_height
                
        box_color = colors[predicted_id]
        #   Thanks to this assignment process, we will draw the RGB codes of
        # the index in the "predicted_ID" from the "colors" assignment we
        # previously made. So we will choose a different color for each label.
        box_color = [int(each) for each in box_color]
        #   We wrote this code to keep these color values in a "list".
                
        lbl = "{}: {:.2f}%".format(lbl, confidence*100)
        #   What we're doing here is a formatting process. So; write "{}",
        # ("lbl") in the first curly braces. In the second set brackets, the
        # format ("{: .2f}", "confidence * 100"), ie percent (%) will be written.
        
        print("predicted object {}".format(lbl))
        
        #   cv2.rectangle(canvas, (starting point), (ending point), color, thickness)
        cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,2)
        
        #   cv2.putText(canvas, what we will print?, (What side will the printed
        # items be on the canvas?), font, font size, text color, thickness)
        cv2.putText(frame,lbl,(start_x,start_y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, box_color, 1)
    
    
    cv2.imshow("Detection Window", frame)
    #   We write to see all the transactions we have done.
    
    #   We make an assignment to shutdown so that we can easily close the code
    # we are running:
    if cv2.waitKey(1) & 0xFF == ord("w"):
        break


md.release()
cv2.destroyAllWindows()
