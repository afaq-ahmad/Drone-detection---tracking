#python darknet_video.py --obj=drone_detection/mavicpro2_drone_dataset/obj.data --cfg=drone_detection/mavicpro2_drone_dataset/yolov3.cfg --weight=drone_detection/mavicpro2_drone_dataset/backup/yolov3_best.weights --trck_len=30 --input=test1.mp4
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import sys
import argparse

############################ Tracking Init############################
from timeit import time
import warnings
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "test1.mp4")
ap.add_argument("-o", "--obj",help="obj file of training", default = "drone_detection/mavicpro2_drone_dataset/obj.data")
ap.add_argument("-c", "--cfg",help="yolov3 config file", default = "drone_detection/mavicpro2_drone_dataset/yolov3.cfg")
ap.add_argument("-w", "--weight",help="weight file path", default = "drone_detection/mavicpro2_drone_dataset/backup/yolov3_best.weights")
ap.add_argument("-tl", "--trck_len",help="Tracking Lenght", default = 50)

args = vars(ap.parse_args())


############################ Tracking Init############################
pts = [deque(maxlen=int(args["trck_len"])) for _ in range(9999)]
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")


netMain = None
metaMain = None
altNames = None




def YOLO():

    global metaMain, netMain, altNames
    metaPath = args["obj"]#"drone_detection/mavicpro2_drone_dataset/obj.data"
    configPath = args["cfg"]#"drone_detection/mavicpro2_drone_dataset/yolov3.cfg"
    weightPath = args["weight"]#"drone_detection/mavicpro2_drone_dataset//backup/yolov3_best.weights"
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args["input"])
    cap.set(3, 1280)
    cap.set(4, 720)
    
    
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    #################################################################
    counter = []
    writeVideo_flag = True
    #Definition of the parameters
    max_cosine_distance = 0.5 #Control threshold of cosine distance
    nms_max_overlap = 0.3 #Non-maximum suppression threshold
    nn_budget = None
    model_filename = 'deep_sort/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    fps = 0.0
    if writeVideo_flag:
        # w = int(cap.get(3))
        # h = int(cap.get(4))
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0,(darknet.network_width(netMain), darknet.network_height(netMain)))
        print("Starting the YOLO loop...")
        list_file = open('detection.txt', 'w')
        frame_index = -1
    
    
    
    while True:
        t1 = time.time()
        ret, frame_read = cap.read()
        if ret != True:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        if len(detections)>0:
            # print(len(detections)) 
            detections=np.array(detections)
            # print('detections1',[detections])
            # print([detections])
            boxs=np.array(detections[:,2].tolist())
            boxs[:,0]=boxs[:,0]-(boxs[:,2]/2)
            boxs[:,1]=boxs[:,1]-(boxs[:,3]/2)
            
            boxs=boxs.astype(int)
            boxs=boxs.tolist()
            # print('boxs1',[boxs])
            class_names=[[cls.decode()] for cls in detections[:,0]]
            
            # print([boxs,class_names])
            ######################################### Tracking #########################################
            features = encoder(frame_resized,boxs)
            # score to 1.0 here).
            # print('boxs2',[boxs])
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # print('detections2',[detections])
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # print('detections3',[detections])
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            i = int(0)
            indexIDs = []
            c = []
            boxes = []
            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame_resized,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                #boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track.track_id))
                counter.append(int(track.track_id))
                bbox = track.to_tlbr()
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

                cv2.rectangle(frame_resized, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
                cv2.putText(frame_resized,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
                if len(class_names) > 0:
                   class_name = class_names[0]
                   cv2.putText(frame_resized, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

                i += 1
                #bbox_center_point(x,y)
                center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
                #track_id[center]
                pts[track.track_id].append(center)
                thickness = 5
                #center point
                cv2.circle(frame_resized,  (center), 1, color, thickness)

            #draw motion path
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame_resized,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                    #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

            count = len(set(counter))
            cv2.putText(frame_resized, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame_resized, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame_resized, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
            cv2.namedWindow("YOLO3_Deep_SORT", 0);
            cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
            cv2.imshow('YOLO3_Deep_SORT', frame_resized)

            if writeVideo_flag:
                #save a frame
                out.write(frame_resized)
                frame_index = frame_index + 1
                list_file.write(str(frame_index)+' ')
                if len(boxs) != 0:
                    for i in range(0,len(boxs)):
                        list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
                list_file.write('\n')
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            #print(set(counter))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
