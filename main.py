"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
import tensorflow as tf
import numpy as np
import datetime

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT =  3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_bboxes(frame, result,width, height, prob_threshold):

    predictions = result[0,0,:,:]
    
    bboxes = predictions[(predictions[:,2]>= prob_threshold)]
    
    current_count = bboxes.shape[0]
   
    for box in bboxes:
        
        xmin = int(box[3] * width)
        ymin = int(box[4] * height)
        xmax = int(box[5] * width)
        ymax = int(box[6] * height)
           
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    
    return frame,current_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    total_count = 0
    prev_total_count = 0
    prev_count = [0]
    enter_scene_time = None
    out_scene_time = None
    person_avg_duration = 0
    vedio_witer = None
    out_path = "output"
    infer_time = 0
    prev_count_mode = 0
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###

    infer_network.load_model(args.model,args.device)

    input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###

    image_flag = False
    
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True

    
    vedio_catpure = cv2.VideoCapture(args.input)

    catpure_width = int(vedio_catpure.get(3))
    catpure_height = int(vedio_catpure.get(4))

    if not image_flag:
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        file_name,file_ext = os.path.splitext(os.path.basename(args.input))
        vedio_witer = cv2.VideoWriter(f'{out_path}/{file_name}.avi',fourcc, 30, (catpure_width,catpure_height))

    ### TODO: Loop until stream is over ###
    while(vedio_catpure.isOpened()):

        ### TODO: Read from the video capture ###
        ret, frame = vedio_catpure.read()

        if  not ret:
            break
        
        ### TODO: Pre-process the image as needed ###
        Pre_process_frame = cv2.resize(frame,(input_shape[3],input_shape[2]))
        Pre_process_frame = Pre_process_frame.transpose((2,0,1))
        Pre_process_frame =  Pre_process_frame.reshape(1, *Pre_process_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(Pre_process_frame)
        
        ### TODO: Wait for the result ###
        if  infer_network.wait() == 0:
            st_time = datetime.datetime.now()
            
            ### TODO: Get the results of the inference request ###
            inference_output = infer_network.get_output()
            
            end_time = datetime.datetime.now()
            
            infer_time += (end_time-st_time).microseconds

            frame,current_count = draw_bboxes(frame, inference_output, catpure_width,catpure_height,prob_threshold)

            ### TODO: Extract any desired stats from the results ###

            try:
                prev_count_mode = mode(prev_count)
            except:
                prev_count_mode = max(prev_count)


            if current_count > prev_count_mode or len(prev_count) == 0:
                total_count = total_count + current_count - prev_count_mode
                enter_scene_time = datetime.datetime.now()

            prev_total_count = total_count

            if current_count == 0 and total_count == prev_total_count:
                out_scene_time = None
            else:
                out_scene_time = datetime.datetime.now()
            

            if len(prev_count) > 30:
                prev_count = [current_count]
                
            prev_count.append(current_count) 
            
            if out_scene_time != None and enter_scene_time != None and out_scene_time  > enter_scene_time:
                person_avg_duration = (out_scene_time - enter_scene_time).seconds
                
            else:
                  person_avg_duration = 0  
            
            cv2.putText(frame, f"Current cnt: {current_count}",(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,2)
            cv2.putText(frame, f"Total cnt: {total_count}",(50, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,2)
            cv2.putText(frame, f"Avg Time: {person_avg_duration}",(50, 150),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,2)

            if image_flag:
                cv2.imwrite(f"{out_path}/{os.path.basename(args.input)}", frame)
            else:
                
                vedio_witer.write(frame)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            #client.publish("person", json.dumps({"count": current_count, "total":total_count}))
            ### Topic "person/duration": key of "duration" ###
            #client.publish("person/duration", json.dumps({"duration": person_avg_duration}))
            
        
        ### TODO: Send the frame to the FFMPEG server ###    
        # sys.stdout.buffer.write(frame)  
        # sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
    
    vedio_witer.release()
    vedio_catpure.release()  
    #client.disconnect() 
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = None# connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
