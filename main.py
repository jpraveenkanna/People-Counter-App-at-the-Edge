#!/usr/bin/env python3
import os
import sys
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import time
import socket
import json
import logging as log
import paho.mqtt.client as mqtt
import time
import logging 

from argparse import ArgumentParser
from inference import Network



#Logging
logging.basicConfig(filename="person_counter.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--Device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.1,
                        help="Probability threshold for detections filtering"
                        "(0.7 by default)")
    parser.add_argument("-t", "--Type_of_media", type=str, default=0.1,
                        help="image or video""(0.7 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def handle_video(self,input_source,prob_threshold,net_shape,client):
    
    client_mqtt = client
    start_signal = False
    in_sec = 0
    x_gradient = 0
    timer = 0
    prev_count = 0
    x_prev = 0
    count = 0
    counter = 0
    center_info =[]
    duration = np.array([])
    avg_duration = 0.0
    total_count_copy = 0

    n, c, h, w = self.network.inputs[self.input_blob].shape
    cap = cv2.VideoCapture(input_source)
    res_width = int(cap.get(3))
    res_height = int(cap.get(4))
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #saving format
    out = cv2.VideoWriter('out20.mp4', fourcc, 10, (res_width,res_height)) #To save video
    n, c, h, w = self.network.inputs[self.input_blob].shape


    while cap.isOpened():
        
        flag, frame = cap.read()
        if not flag:
            break

        image_copy = frame
        image = frame
        image_shape = (image.shape[1], image.shape[0])
        frame , center= self.infer_on_track_image(image,image_shape,net_shape,prob_threshold) #Running inference
        counter+=1



        #Finding out new person entry
        if((prev_count==count) and (prev_count != 0) and start_signal ):
            timer +=1
            if len(center)==1:
                
                if(center[0][1] > 690): 
                    #690 ---> x axis exit value
                    start_signal = False
                    duration = np.append(duration,round(in_sec, 1))
                    avg_duration = np.average(duration)
                    client_mqtt.publish('person/duration',payload=json.dumps({'duration': round(duration[-1])}),qos=0, retain=False)
                    timer = 0
                    


        in_sec = timer * 0.09971 #Calculating time for each frame in seconds


        if len(center)==1: #Tracking person if person exist
            try:
                if(center[0][1]):
                    if(x_prev == 0):
                        x_prev = center[0][1]
                        count+=1
                    else:
                        x_gradient = abs(x_prev - center[0][1]) 
                        x_prev = center[0][1]
                        if(x_gradient>150):
                            count+=1
               
            except:
                logger.info("Exception in tracking person") 
                
            
            center_info.append(x_gradient)

            cv2.putText(frame, "Total Person Counted:"+str(count), (15, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, "Persons on Screen:"+str(int(start_signal)), (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, "Duration :"+str(round(in_sec, 1))+" Second", (15, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, "Avg Duration :"+str(round(avg_duration,1))+" Second", (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            total_count_copy = int(count)
            out.write(frame)
            try:
                client_mqtt.publish('person',payload=json.dumps({'count': int(start_signal), 'total': total_count_copy}),qos=0, retain=False)
                frame_stream = cv2.resize(frame, (668, 432))     
                sys.stdout.buffer.write(frame_stream)
                sys.stdout.flush()
                
            except:
                logger.info("Exception in MQTT of ffmpeg server")

             

        else:
            
            cv2.putText(image_copy, "Total Person Counted:"+str(count), (15, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(image_copy, "Persons on Screen:"+str(int(start_signal)), (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(image_copy, "Duration :"+str(round(in_sec, 1))+" Second", (15, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(image_copy, "Avg Duration :"+str(round(avg_duration,1))+" Second", (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            total_count_copy = int(count)
            out.write(image_copy)
            try:
                client_mqtt.publish('person',payload=json.dumps({'count': int(start_signal), 'total': total_count_copy}),qos=0, retain=False)
                frame_stream = cv2.resize(image_copy, (668, 432))     
                sys.stdout.buffer.write(frame_stream)
                       
                sys.stdout.flush()
                
            except:
                logger.info("Exception in MQTT of ffmpeg server")
            
            

        #Tracking change in person count
        
        if((prev_count < count) and (prev_count != count)):
            prev_count = count
            start_signal = True

       
        
        
        key = cv2.waitKey(30)
        esc_code = 27
        if key == esc_code:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    client_mqtt.disconnect()
    

def handle_image(self,input_source,prob_threshold,net_shape,client):
    client_mqtt=client
    image = cv2.imread(input_source)
    image_shape = (image.shape[1], image.shape[0])

    result, center = self.infer_on_track_image(image,image_shape,net_shape,prob_threshold)

    frame_stream = cv2.resize(result, (768, 432))
    #cv2.imwrite("output1.jpeg", result)     
    sys.stdout.buffer.write(frame_stream)
    
    client_mqtt.publish('person',payload=json.dumps({'count': len(center), 'total': len(center)}),qos=0, retain=False)
    client_mqtt.publish('person/duration',payload=json.dumps({'duration': 0}),qos=0, retain=False)
    client_mqtt.disconnect()
    sys.stdout.flush()
    
    return True

def infer_on_stream(net,args, client,net_shape):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = net
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    media_type =args.Type_of_media
    input_source = args.input
    
    if(media_type == "image"):
        handle_image(infer_network,input_source,prob_threshold,net_shape,client)
        
    elif(media_type == "video"):
        handle_video(infer_network,input_source,prob_threshold,net_shape,client)
        
    else:
        print("Type video or image")
        
         
        
        
        
   


def main():

    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    model = args.model
    Device = args.Device
    CPU_extension = args.cpu_extension
    net = Network()
    net.load_model(model,Device,CPU_extension)
    
    net_input_shape = net.get_input_shape()['image_tensor']
    net_shape = (net_input_shape[3], net_input_shape[2])
    infer_on_stream(net,args, client,net_shape)


if __name__ == '__main__':
    main()