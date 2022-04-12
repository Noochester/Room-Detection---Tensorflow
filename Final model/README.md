
# Final model for the Room Object Detection project

Utilizing EfficientDet D2 768x768 and TensorFlow\
Trained with NVIDIA 2080TI 11GB GPU\
Training was done thanks to:\
https://github.com/wangermeng2021/EfficientDet-tensorflow2

The dataset images were labeled with https://github.com/tzutalin/labelImg

**Collect dataset and pre-trained model files from here:** \
-The images the model was trained on\
https://drive.google.com/drive/folders/1EYnM3vn9E0Dr5KBjAe1PKo0CVZ65At8R?usp=sharing 
\
-Model\
https://drive.google.com/drive/folders/18FYiqiDYktOS8FVQ6hW4lqKMM8nlYJAx?usp=sharing

# Description
The current model is trained to recognize and draw boxes around the following object types:\
-**chair**, **table**, **bed**, **door**, **pictureOrTV**, **Wheelchair/rollator**\
When *pictureOrTV* is detected, it is masked, trying to prevent these objects being recognized as people. 
The model has been trained on a relatively small dataset(~1000images) because of hardware limitations, therefore the results it produces can be very inaccurate at times. The accurate object detections are generally in the 0.2-0.6 threshold.

# Instructions
**Install TensorFlow API** (skip if you have TensorFlow built)

The following instructions are for Python 3.9 and Windows OS:
```
mkdir Tensorflow
cd Tensorflow
```
*-(optional)* Create a virtual env or skip step to install globally
```
pip install virtualenv
virtualenv --python=3.9 tensorvenv
.\tensorvenv\Scripts\activate
```
-Collects the packages and install
```
git clone https://github.com/tensorflow/models
pip install wget
python
>>>import wget
>>>url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
>>>wget.download(url)
>>>exit()

tar -xf protoc-3.15.6-win64.zip
set PATH=%PATH%; *full path*\Tensorflow\bin
``` 
``` PATH``` (check if last line is the correct path pointing to the bin path)

Install TensorFlow with Protocol buffers
```
cd Tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
copy object_detection\\packages\\tf2\\setup.py setup.py
python setup.py build
python setup.py install
pip install -e .
```

**What you will see next is dependencies shown in red, install all required packages with pip.**\
```pip install pycocotools``` will require MSVC 14.0+ installed\
```pip install tensorflow-cpu``` Install this as well, it utilizes the CPU, which is enough for detections and doesn't require installation of additional libraries (CUDA and cuDNN)

**Verify installation with:**
```python``` *your path*``` \Tensorflow\models\research\object_detection\builders\model_builder_tf2_test.py```

If you get **OK (skipped=1)** the installation is finished.
![Example1](Example2.png?raw=true "Verification")


**Running the detection**
```
git clone https://github.com/Noochester/Room-Detection---Tensorflow.git
cd Room-Detection---Tensorflow/Final model
```
**Collect the model files from the link at the top, so you now have "\Final model\export"**

The script can be adjusted to detect different amount of boxes, with different colours, fonts and thickness, set amount of the different objects. By default it checks the **/images/eval** folder for images and the results are saved in the **images/evald**.\
**pictureOrTV** is by default 0 so the value must be changed with the arguments to get detections of it. 

An example command to run the script:
```
python detect.py --model-dir export/best_model_d2_112_0.655/2 --pic-dir images/eval --class-names dataset/proj.names --score-threshold 0.2
```
```deactivate``` Disables the virtual enviroment, use the activate command to use it again.

The output should be the x1y1 and x2y2 coordinates, the score and the object name. Example below:
![Example2](Example1.png?raw=true "Outputs")
