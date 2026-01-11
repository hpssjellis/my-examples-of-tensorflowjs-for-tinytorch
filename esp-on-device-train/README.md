Started Jan 10, 2026

This folder has the best so far of the programs for on device [XIAO ML kit](https://www.seeedstudio.com/The-XIAOML-Kit.html) training. Another folder has the raw drafts.


## 3 programs to do it all

### Overview

1. Program 1 takes pictures,  try 30 distinct images for each class. Just hold A0 to trigger the image taking
2. Program 2 trains the 2 convolution layer vision model, exports a header file (and a binary copy) To train just hold A0
3. Program 3 does  bare metal inference no TFLITE needed. For debug information just hold A0 

### micro sd card structure:

/images/CLASS0_NAME/.jpg      (For this code, it needs three folders with the label names as the folder names)  
/images/CLASS1_NAME/.jpg    
/images/CLASS2_NAME/.jpg        

/header/myModel.h                (Generated or reloaded by program 2)

/header/myWeights.bin            (Generated or reloaded by program 2)

Just make the images and header folders the code will generate the header files. I have included a header folder in this github so you can see the format.
Reminder you have to move the images from program 1 into the correct class named folders in the images folder.

### Detaied steps

## a01-esp23s3-image-capture.ino  

1. On trigger (squeeze A0) captures images every 100 ms
2. Manually you move these images into an "images" folder with three folders for each of the different training class
3. Folder names are the class names.
4. Only load 30 images per class. At the moment i am not sure why there is this limit, the PSRAM has much more space available.

## a02-esp32s3-train.ino

1. loads images, checks header folder for a myWeights.bin file, else random sets weights.
2. on A0 trigger trains one batch, prints debug and saves the myModel.h file and a binary myWeights.bin file to disk in the "header" folder.
3. Note the binary file is just an easier, cleaner way for the mcu to load the weights for another round of training

## a03-esp32s3-inference.ino

1. runs inference, this is the code you would mess with for your project
2. Must manually move the myModel.h include file to this code. (Must know how to make a new tab in the Arduino IDE)
3. A0 trigger prints debug info


Still to do: is to make things easier to train using the webpage [torchjs00.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs00.html)

Note: myModel.h and myWeights.bin are the same fill in different formats. They do not work together. myModel.h is human readable to help spot issues. 
myWeights.bin is cleaner and easier for the micrcontroller to load at the start of a training session. The inference program uses myModel.h



## Student Learning

This code is not to make the best Vision model it is to help students understand why they use edgeImpulse and TFLITE, while giving them full local control of the entire pipleine.






