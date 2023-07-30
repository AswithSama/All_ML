import cv2 as cv
import numpy as np
import message
import math
import time as time
video=cv.VideoCapture('counting_test_case 1 and 2.mp4')
video.set(cv.CAP_PROP_FPS,int(10))
fps=video.get(cv.CAP_PROP_FPS)
#print(fps)
with open('coco.names') as f:
    labels=[line.strip() for line in f]
network=cv.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
layersNames=network.getLayerNames()
layersOutput=[layersNames[i-1] for i in network.getUnconnectedOutLayers()]
prob_min=0.5
threshold=0.3
colours=np.random.randint(0,255,size=(len(labels),3),dtype='uint8')
classification=[0]*4
frames_with_traffic=0
lifes=0
no_of_frames=0
center_points_prev_frame=[]
tracking_objects={}
object_id=0
car_count=0
TCD_using_tracking=[]
bike_count=0
bus_count=0
truck_count=0
count_list=[]
start_time=0
end_time=0
id1=0
max1=-200
total_time=0
counting_number_of_vehicles=0
no=0
flag2=False
main=[]
for_counting_unique_vehicles=[]
while True:
    no_of_frames+=1
    flag=0
    center_points_cur_frame=[]
    ret,frame=video.read()
    if not ret:
        end_time=time.time()
        break 
    cv.namedWindow('video',cv.WINDOW_NORMAL)
    blob=cv.dnn.blobFromImage(frame,1/255,(416,416))
    y_c=frame.shape[0]//2
    network.setInput(blob)
    #start=t.time()
    output_from_network=network.forward(layersOutput)
    bounding_boxes=[]
    confidences=[]
    class_numbers=[]
    c=[]
    id=[]
    y=frame.shape[0]-100
    #FINDING THE CENTER POINT CO-ORDINATES OF VEHICLES,CONFIDENCES,CLASS_NUMBERS(TYPE OF VEHICLE).
    for results in output_from_network:
        for detectedObjects in results:
            scores=detectedObjects[5:]
            class_current=np.argmax(scores)
            confidence_current=scores[class_current]
            if confidence_current>prob_min:
                box_current=detectedObjects[0:4]*np.array([frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]])
                x_center,y_center,box_width,box_height=box_current
                x_min=int(x_center-(box_width/2))
                y_min=int(y_center-(box_height/2))
                bounding_boxes.append([x_min,y_min,int(box_width),int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
                #print(detectedObjects[0:4])
    results=cv.dnn.NMSBoxes(bounding_boxes,confidences,prob_min,threshold)
    no_of_vehicles=len(results)
    #FINDING ALL OF THE ABOVE MENTIONED ENDS HERE
    #PREDICTING TRAFFIC CONGESTION STARTS HERE
    #print(no_of_vehicles)
    if id1 in tracking_objects:
        flag2=True
    else:
        flag2=False
    #no_of_vehicles>30 or
    if(no_of_vehicles>25 or flag2):
        frames_with_traffic+=1
        flag=1
    if(flag==0 and frames_with_traffic>0):
        lifes+=1
        if(lifes>1):
            frames_with_traffic=0
            lifes=0
    if(frames_with_traffic>150):
        message.Message()
        print("There is a traffic congestion in the video")
        break
        frames_with_traffic=0
    #PREDICTING TRAFFIC CONGESTION ENDS HERE
    #DRAWING THE BOUNDING BOXES AND LABELING THE VEHICLES
    '''if(no_of_frames>150):
    print(no_of_frames)'''

    if len(results)>0:
        for i in results.flatten():
            if(int(class_numbers[i])==2) or (int(class_numbers[i])==3) or (int(class_numbers[i])==5) or (int(class_numbers[i])==7):
                x_min,y_min=bounding_boxes[i][0],bounding_boxes[i][1]
                box_width, box_height=bounding_boxes[i][2],bounding_boxes[i][3]
                x_center=x_min+int(box_width/2)
                y_center=y_min+int(box_height/2)
                color_box_current=colours[class_numbers[i]].tolist()
                cv.rectangle(frame,(x_min,y_min),(x_min+box_width,y_min+box_height),color_box_current,2)
                cv.circle(frame,(x_center,y_center),4,(0,0,255),-1)
                center_points_cur_frame.append((x_center,y_center))
                for_counting_unique_vehicles.append(((x_center,y_center),int(class_numbers[i])))
                #print(for_counting_unique_vehicles)
                id.append(int(class_numbers[i]))
                text_box_current='{}'.format(labels[int(class_numbers[i])])
                cv.putText(frame,text_box_current,(x_min-5,y_min),cv.FONT_HERSHEY_COMPLEX,0.7,color_box_current,2)
                #print(y_center,y)
    #WHOLE DETECING PART ENDS HERE

#TRACKING PART STARTS HERE
        if no_of_frames<=2:
            for pt in center_points_prev_frame:
                for pt2 in center_points_cur_frame:
                    distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                    if distance<30:
                        tracking_objects[object_id]=pt
                        object_id+=1
                        #print(object_id)
        
        else:
            tracking_objects_copy=tracking_objects.copy()
            center_points_cur_frame_copy=center_points_cur_frame.copy()
            for object_id,pt2 in tracking_objects_copy.items():
                object_existence=False
                for pt in center_points_cur_frame_copy:
                    distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                    if distance<20:
                        tracking_objects[object_id]=pt
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        object_existence=True
                        continue
                if not object_existence:
                    tracking_objects.pop(object_id)
            for pt in center_points_cur_frame:
                tracking_objects[object_id]=pt
                object_id+=1
                #print(object_id)
                
                
        for object_id,pt in tracking_objects.items():
            cv.putText(frame,str(object_id),pt,cv.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),1)
#TRACKING PART ENDS HERE (95%)
        #COUNTING STARTS HERE
        cv.line(frame,(0,frame.shape[0]-200),(frame.shape[1],frame.shape[0]-200),(0,0,255),2)
        y_coordinate=frame.shape[0]-200
        for object_id, pt in tracking_objects.items():
            distance2=pt[1]-y_coordinate
            if distance2<10 and distance2>-10:
                if object_id not in count_list:
                    counting_number_of_vehicles+=1
                    for i in for_counting_unique_vehicles:
                        if pt==i[0]:
                            if i[1]==3:
                                bike_count+=1
                            elif i[1]==2:
                                car_count+=1
                            elif i[1]==5:
                                bus_count+=1
                            else:
                                truck_count+=1
                    count_list.append(object_id)
                cv.line(frame,(0,frame.shape[0]-200),(frame.shape[1],frame.shape[0]-200),(0,255,0),2)
        #COUNTING ENDS HERE
        #traffic congestion based on tracking starts here
        key_list=list(tracking_objects.keys())
        val_list=list(tracking_objects.values())
        if no_of_frames==2:
            for i in for_counting_unique_vehicles:
                if(i[1]==7 or i[1]==5):
                    distance3=y_coordinate-i[0][1]
                    TCD_using_tracking.append((i[0],distance3))
                    if max1<distance3:
                        max1=distance3
                        id1=val_list.index(i[0])
            #print(id1)
            
        
        '''print(for_counting_unique_vehicles)
        print("\n",TCD_using_tracking)'''

        #traffic congestion based on tracking ends here
        #print(center_points_cur_frame)
        #i=0
        #print(tracking_objects,id)
        '''for object_id,pt in tracking_objects.items():
                main.append((id[i],object_id,pt))
                i+=1
            print(main)'''
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.namedWindow('Count',cv.WINDOW_NORMAL)
    cv.putText(blank,'Number Of Cars: '+str(car_count),(30,30),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
    cv.putText(blank,'Number Of Bikes: '+str(bike_count),(30,70),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
    cv.putText(blank,'Number Of Buses: '+str(bus_count),(30,110),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
    cv.putText(blank,'Number Of Trucks: '+str(truck_count),(30,140),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
    cv.namedWindow('count',cv.WINDOW_NORMAL)
    cv.imshow('count',blank)
    cv.namedWindow('frame',cv.WINDOW_NORMAL)                      
    cv.imshow('frame',frame)
    #print(no_of_vehicles)
    center_points_prev_frame=center_points_cur_frame.copy()
    #print(detectedObjects[0:4])
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
#print(no_of_frames//fps)
#print(count_list)
#print(no_of_frames)
'''print(counting_number_of_vehicles)
print((end_time-start_time)*no_of_frames)
print((end_time-start_time))'''
cv.destroyAllWindows()
#print(len(count))
'''cv.imshow('video',frame)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
cv.destroyAllWindows()        

for i in range(len(c))
        if (c[i]<y+2 and c[i]>y-2):
            if( id[i]==2):
                classification[0]+=1
                #print('car:',classification[0])
            if( id[i]==3):
                classification[1]+=1
                #print('Motor Bike:',classification[1])
            if( id[i]==5):
                classification[2]+=1
                #print('Bus:',classification[2])
            if( id[i]==7):
                classification[3]+=1
                #print('Truck:',classification[3])
            cv.line(frame,(0,frame.shape[0]-100),(frame.shape[1]-100,frame.shape[0]-100),(0,255,0),2)
            #count.append("vehicle")
            #print(count)'''


'''  i=0
        cv.line(frame,(0,y_c),(frame.shape[1],y_c),(0,0,255),2)
        for object_id,pt in tracking_objects.items():
            distance2=y_c-pt[1] 
            if (distance2<6 and distance2>-6):
                #print(distance2)
                if(id[i]==2):
                    car_id_set.add(object_id)
                    print("no.of.cars",len(car_id_set))
                if(id[i]==3):
                    bike_id_set.add(object_id)
                    print("no.of. bikes",len(bike_id_set))
                if(id[i]==5):
                    bus_id_set.add(object_id)
                    print("no.of.buses",len(bus_id_set))
                if(id[i]==7):
                    truck_id_set.add(object_id)
                    print("no.of.trucks",len(truck_id_set))'''
