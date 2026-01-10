Started Jan 10, 2026

This folder has the best so far programs for on device [XIAO ML kit](https://www.seeedstudio.com/The-XIAOML-Kit.html) training.


sd card structure
/images/<label_name>/*.jpg      (For this code needs three folders with the label names sorted by name)
/header/myModel.h
/header/myWeights.bin


presently we have three propgrams:

## First ino file a01-esp23s3-image-capture.ino  On trigger (squeeze A0) captures images every 100 ms

1. Manually you move images to an image folder with three folders for a different training class
2. Folder names are the class names.

## Second ino file  a02-esp32s3-train.ino

1. loads images, checks header folder for a myWeights.bin file
2. on A0 trigger trains one batch, prints debug and saves the myModel.h file and a binary weight.bin file to disk in the headers folder.
3. Note the binary file is just an easier, cleaner way to do load the weights.

## third program  a03-esp32s3-inference.ino

1. runs inference, this is the code you would mess with for your project
2. Must manually move the myModel.h include file to this code. (Must know how to make a new tab in the Arduino IDE)
3. A) trigger prints debug info


Still to do is to make things easier to train using the webpage [torchjs00.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs00.html)

Note: myModel.h and myWeights.bin are the same fill in different formats. They do not work together. myModel.h is human readable to help spot issues. 
myWeights.bin is cleaner and easier for the micrcontroller to load at the start of a training session.

