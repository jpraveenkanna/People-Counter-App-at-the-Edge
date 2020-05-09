# People Counter App at the Edge
The people counter application will demonstrate a smart video IoT solution. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

<h2><a  class="anchor" aria-hidden="true" ><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Software Dependencies</h2>

<ul>
<li>
<p><b>OpenVINO toolkit 2019 R3</b> - To run infereance on deeplearning model</p>
</li>
<li>
<p><b>OpenCL</b> - To run on GPU (Application tested on Intel Integrated Graphics)</p>
</li>
<li>
<p><b>Node server</b> - For UI which displays video frame and statistics information from MQTT server</p>
</li>
<li>
<p><b>FFMPEG 3.4 server</b> - For streaming video frames realtime</p>
</li>
<li>
<p><b>MQTT Mosca server</b> - For feeding processed data to the UI server.</p>
</li>
</ul>

<h2><a  class="anchor" aria-hidden="true" ><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Model Used</h2>
<p>
  The model is choosen from Tensorflow Open Model Zoo
  <ul>
  <li>
    <p><a href="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"> SSD Mobilenet V2 trained on COCO dataset </a></p>
</li></ul>

To run the model at edge. Tensorflow model is converted Intermediate Representation (IR File). This IR file is loaded to openvino inference engine for inference.
<br><h5>Tensorflow Model to IR file - Conversion command:-</h5>
<pre><code>
sudo python /opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/mo_tf.py \
--input_model='ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb' \
--tensorflow_use_custom_operations_config '/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json' \
--tensorflow_object_detection_api_pipeline_config 'ssd_mobilenet_v2_coco_2018_03_29/pipeline.config' \
--reverse_input_channels
</code></pre><br>
  </p>
<p>Normally a deep learning model will have multiple layers each having its own activation funtion. These activation funtion are mathematical operations which will run on computation hadware like CPU or hardware accelerators like GPU,TPU,Intel Neural Stick. All operations are normally not supported by all the hardwares. </p>

<p>For that purpose Openvino have custom layers which converts unsupported layers and also optimize the operation to run efficiently on specified hardware.</p>

<p>Openvino have <code>query_network()</code> function which checks the supported layers and have pre-defined library extensions which can be loaded to the machine using <code>add_extension()</code> function.  Once all the layers are supported then it can be used to infer the model. </p>

<p>One main advantage of using custom layers is that the model is optimised and will give consistant performance and better efficiency.</p> 

<h2><a  class="anchor" aria-hidden="true" ><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Comparing Model Performance</h2>

<p>
  <h4>Size of the model:</h4>
    <ul>
  <li>Tensorflow Model  -----> 69.7 MB     </li>
  <li>OpenVino IR file  -----> 67.4 MB     </li>
    </ul>
 </p>
 
 <p>
  <h4>Average Inference Time:</h4>
    <ul>
  <li>Tensorflow -----> 218 ms     </li>
  <li>OpenVino   -----> 153 ms     </li>
    </ul>
 Average inference value of 994 video frames performed on the same hardware. 
 </p>
 
  <p>
  <h4>Accuracy:</h4>
   Openvino could able to detect more objects in some cases and the probability of the outputs are same for both tensorflow and openvino.
   
 </p>
 
 
 <h2><a  class="anchor" aria-hidden="true" ><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Potential Model Usecase</h2>
 <p>
1. To ugprade an existing road or to construct a new road, road survey officers spend weeks onsite to record how many vechicles and the type of vechicle passing the road. Using the information they calculate the load/stress that the road need to withstand for a period of time. Using this model we can get much accurate result on vechicle details on road. A portable IOT device can be made for this purpose which will stream real time data.<br><br>2. In a large scale vechicle parking lots like mall parking, office vehicle parking, bus depots. This model can be used to  monitoring vehicles entering and leaving the gate. Using that it can be used to calculate free parking space realtime.<br><br>3. This model can be used to calculate average time and trend on unoccupied common office meeting rooms.
</p>

 <h2><a  class="anchor" aria-hidden="true" ><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Potential Effects of the Model</h2>
 
 <p>
<b>Lighting:</b><br> Dark or too bright images will be difficult for a model to recognise a object. Some times it may misclassify objects. For realtime CCTV camera like application. placement of camera is very important. Good lignting condition can  improve the results greatly.<br><br>

<b>Camera focal length: </b> <br>Camera focal length decides the quality of the image which inturn helps model to give better results. Blur images are difficult to detect and may misclassify<br><br>

<b>Image size: </b><br>If the object is too close. I will be difficult to detect. And even if it detect the probability of the object will decrease steeply. If object is far away then the convolution layer reduces the image size and filters out the data. The model can only accept particular size. So if the image supplied to the model is close to this size then it will produce better result. 
</p>
