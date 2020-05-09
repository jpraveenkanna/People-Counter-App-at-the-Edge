#!/usr/bin/env python3
import os
import sys
from openvino.inference_engine import IENetwork, IECore
import cv2
#from matplotlib import pyplot as plt
import numpy as np
import time

class Network:

    def __init__(self):
        
        self.network = None
        self.plugin = None
        self.exec_network = None
        self.infer_request_handle = None
        self.input_blob = None
        self.output_blob = None
        
        
        
    def check_layers(self,plugin, network):
        layers_supported = plugin.query_network(network, device_name='CPU')
        layers_in_model = network.layers.keys()
        all_layers_supported = True
        for l in layers_in_model:
            if l  not in layers_supported:
                all_layers_supported = False
                #print('Layer', l, 'is not supported')
        if all_layers_supported:
            #print('All layers supported')
            pass
        return all_layers_supported
        

    def load_model(self,model,Device,CPU_extension):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        Is_all_layers_supported = self.check_layers(self.plugin, self.network)
        if not Is_all_layers_supported:
            #print("Adding extension:", CPU_extension)
            self.plugin.add_extension(CPU_extension, "CPU")
            Is_all_layers_supported = self.check_layers(self.plugin, self.network)
            if not Is_all_layers_supported:
                sys.exit(1)
            else:
                self.exec_network = self.plugin.load_network(self.network, Device)
                self.input_blob = next(iter(self.network.inputs))
                self.output_blob = next(iter(self.network.outputs))
                #print("Sucessfully loaded the model")
                
        return None

    def get_input_shape(self):
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
        return input_shapes

    def exec_net(self,net_input):
        infer_request_handle = self.exec_network.start_async(request_id=0,inputs=net_input)
        return infer_request_handle

    def wait(self):
        status = self.infer_request_handle.wait()
        return status
   
    

    def infer_on_track_image(self,input,image_shape, net_shape,prob_threshold):


        n, c, h, w = self.network.inputs[self.input_blob].shape
        images = np.ndarray(shape=(n, c, h, w))
        images_hw = []
        res =None

        for i in range(n):
            image = input
            ih, iw = image.shape[:-1]
            images_hw.append((ih, iw))
            if (ih, iw) != (h, w):
                image = cv2.resize(image, (w, h))

            image = image.transpose((2, 0, 1))  
            images[i] = image
            net_input = {'image_tensor': images[i]}
            self.infer_request_handle = self.exec_net(net_input) 
            status = self.wait()

            if status == 0:
                inf_start = time.time()
                res = self.infer_request_handle.outputs[self.output_blob]
                inf_end = time.time()
                det_time = inf_end - inf_start
                
            else:
                #print("Can not infer from the frame")
                pass

        data = res[0][0]
        boxes, classes ,probability = {}, {},{}

        for number, proposal in enumerate(data):
                if proposal[2] > 0:
                    imid = np.int(proposal[0])
                    ih, iw = images_hw[imid]
                    label = np.int(proposal[1])
                    confidence = proposal[2]
                    xmin = np.int(iw * proposal[3])
                    ymin = np.int(ih * proposal[4])
                    xmax = np.int(iw * proposal[5])
                    ymax = np.int(ih * proposal[6])
                    
                    if label == 1:

                        if proposal[2] > prob_threshold:
                          
                            if not imid in boxes.keys():
                                boxes[imid] = []
                            boxes[imid].append([xmin, ymin, xmax, ymax])
                            if not imid in classes.keys():
                                classes[imid] = []
                            classes[imid].append(label)
                            if not imid in probability.keys():
                                probability[imid] = []
                            probability[imid].append(proposal[2])
                        

        tmp_image = None
        center_tracking = []

        for imid in classes:
                
                tmp_image = input
                for box,label,prob in zip(boxes[imid],classes[imid],probability[imid]):
                    cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
                    mid_x,mid_y =  (box[2]+(box[0]-box[2])//2),(box[3]+(box[1]-box[3])//2)
                    cv2.putText(tmp_image, str("Person")+"->Prob:"+str(round(prob, 2)), (mid_x,mid_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 250, 234), 1)
                    inf_time_message = 'Inference time: {:.3f} ms'.format(det_time * 1000)
                    cv2.putText(tmp_image, inf_time_message, (15, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

                    center_tracking.append([imid,mid_x,mid_y])

        return tmp_image,center_tracking


