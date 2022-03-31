# Final model for the Room Object Detection project

Utilizing EfficientDet D2 768x768 and TensorFlow

**Collect dataset and pre-trained model files from here:** \
-https://drive.google.com/drive/folders/1EYnM3vn9E0Dr5KBjAe1PKo0CVZ65At8R?usp=sharing
\
-https://drive.google.com/drive/folders/18FYiqiDYktOS8FVQ6hW4lqKMM8nlYJAx?usp=sharing

# Description
The current model is trained to recognize and draw boxes around the following object types:\
-**chair**, **table**, **bed**, **door**, **pictureOrTV**, **Wheelchair/rollator**\
When *pictureOrTV* is detected, it is masked, trying to prevent these objects being recognized as people. 
The model has been trained on a relatively small dataset because of hardware limitations, therefore the results it produces 
can be very inaccurate at times. For the same reason it is using the very low treshold of 0.1.

# Instructions
The script can be adjusted to detect different amount of boxes, with different colours, fonts and thickness. It collects data from
the **/images/eval** folder and the results are saved in the **images/evald**\
An example command to run the script:\
    ```
    python detect.py --model-dir export/best_model_d2_112_0.655/2 --pic-dir images/eval --class-names dataset/proj.names --score-threshold 0.1
    ```

![Example](example1.png?raw=true "Optional Title")
