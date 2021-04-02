# -*- coding: utf-8 -*-

#   Tek bir python sayfamizin icerisini bolumlere ayirmak icin  " #%% "
# isaretini kullaniriz!!!

#   .append()  ::  Listelerin SONUNA eleman eklememizi saglayan metoddur.
#   Threshold  =  Eşik.
#   Güven Skoru  =  Bounding Box'un Confidence degeri.


#   Bu bolumde, bir onceki  " 1_yolo_pretrained_image "  bolumunde 
# yaptigimiz islemin OUT kismindaki resimde, bir nesneyi defalarca
# kez dikdortgen icine almasi problemini cozecegiz. Ve bu problemleri
# NON-MAXIMUM SUPPRESSION bulumunde halledecegiz:
    

import cv2
import numpy as np

img = cv2.imread("C:\YOLO_Mask_Detection\img.jpg")
# print(img)

img_width = img.shape[1]
img_height = img.shape[0]



img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)

labels = ["good", "bad"]

colors = ["0,255,255","0,0,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))



model = cv2.dnn.readNetFromDarknet("C:\YOLO_Mask_Detection\yolov3_mask.cfg", "C:\YOLO_Mask_Detection\yolov3_mask_last.weights")

layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)


############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################
#   *NOTE*  ==  predicted_ID = ids
ids_list = []
boxes_list = []
confidences_list = []

############################ END OF OPERATION 1 ########################



for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.20:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))
            
            
            ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################
            
            #   Burada, yukaridaki "operation 1"'de actigimiz bos listeleri
            # dolduracagiz.
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
            
            ############################ END OF OPERATION 2 ########################
            
            
            
############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################
#   ****NOTE****
#   Asagidaki  " cv2.dnn.NMSBoxes() "  fonksiyonu, EN YUKSEK guvenirlige sahip
# dikdortgenlerin ID'lerini donduruyor. Yani, MAXIMUM CONFIDENCE'ye sahip olan
# Bounding Box'lari bir array biciminde donduruyor:
# cv2.dnn.NMSBoxes(bounding box'larin listesi, guven listesinin listesi, guven skoru, Threshold)
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
#   Su an, "max_ids" icerisinde, en yuksek guvenilir skoruna sahip dikdortgenlerin
# ID'leri var!
 
for max_id in max_ids:
    
    #   Asagida, "max_ids" icerisindeki bir degere, index ile erisiyoruz: 
    max_class_id = max_id[0]
    
    #   Asagida da, "boxes_list" icerisinden "nesneye ait olan box degerlerine"
    # erisecegiz:
    box = boxes_list[max_class_id]
    #   Bu "box" degiskeni, nesneye ait olan bounding_box'un; baslangic noktasi,
    # bitis noktasi, eni, boyu gibi degerleri sirasi ile tutuyor olacak:
    start_x = box[0] 
    start_y = box[1] 
    box_width = box[2] 
    box_height = box[3] 
    
    #   Asagida, "ids_list" icerisinden "tahmin edilen degerlere" tek tek
    # erisecegiz.
    predicted_id = ids_list[max_class_id]
    
    #   "predicted_id" degiskenin icerisinde tuttugu index degeri ile "labels"
    # icerisinden "detect" ettigimiz label'e eristik:
    label = labels[predicted_id]
    
    #   Yine, "max_class_id" degiskenini kullanarak "confidences_list" icerisinden
    # ilgili "label"'imizin sahip oldugu confidence degerine eristik:
    confidence = confidences_list[max_class_id]
  
############################ END OF OPERATION 3 ########################
            


    end_x = start_x + box_width
    end_y = start_y + box_height
            
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
            
    label = "{}: {:.2f}%".format(label, confidence*100)
    print("predicted object {}".format(label))
     
    cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)
    cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)


cv2.imshow("Detection Window", img)
#   Bir onceki calisma sayfamiz olan  " 1_yolo_pretrained_image "  sayfasini da
# calistirirsak, aradaki farki ve duzeltmelerimizi anlayabiliriz.