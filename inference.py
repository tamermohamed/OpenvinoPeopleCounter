#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore,IEPlugin
import cv2

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        
        self.plugin = IECore()
        self.exec_network = None
        self.net_plugin = None
        self.input_blob = None
        self.output_blop = None
        self.infer_request_handle = None

    def load_model(self,path_to_xml_file,device = "CPU"):
       
        path_to_bin_file = os.path.splitext(path_to_xml_file)[0]+'.bin'

        self.net_plugin = self.plugin.read_network(model=path_to_xml_file, weights=path_to_bin_file)
       
        self.exec_network = self.plugin.load_network(self.net_plugin, device)
        
        self.input_blob = next(iter(self.net_plugin.inputs))
        self.output_blop = next(iter(self.net_plugin.outputs))
       
        return self.exec_network

    def get_input_shape(self):
         
        return self.net_plugin.inputs[self.input_blob].shape

    def exec_net(self,image):

        self.infer_request_handle  = self.exec_network.start_async(request_id = 0, inputs = {self.input_blob: image})
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###

        return self.infer_request_handle.wait()
       

    def get_output(self):
        result = self.infer_request_handle.outputs[self.output_blop]
       
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return result