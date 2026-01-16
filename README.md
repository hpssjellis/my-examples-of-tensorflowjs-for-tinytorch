# my-examples-of-tensorflowjs-for-tinytorch


This page as a web page index.

[hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/) <br>

and the Github it is built at   [my-examples-of-tensorflowjs-for-tinytorch](https://github.com/hpssjellis/my-examples-of-tensorflowjs-for-tinytorch)  <br>

This page supports the [mlsysbook.ai](https://mlsysbook.ai/)  and it's tinyTorch implementation at [mlsysbook.ai/tinytorch/intro.html](https://mlsysbook.ai/tinytorch/intro.html)  <br>

This page is trying to simplify for High School students and non-engineering undergrads TinyTorch without needing to do any Python installation<br>

It does not reach the same extent of total understanding of the system, but is a half way point between just using machine learning and TinyTorch understanding. <br>


<h1>Reminder to donate to the openCollective!</h1>

[opencollective.com/mlsysbook](https://opencollective.com/mlsysbook) <br>

[<img width="489" height="477" alt="image" src="https://github.com/user-attachments/assets/a2e3dc57-f9cc-45ee-a998-91cf55509038" />](https://opencollective.com/mlsysbook)


<h3>ML  map</h3>


<img width="1024"  alt="image" src="https://github.com/user-attachments/assets/2247019e-524d-4cc0-8087-b717466cebd3" />









## What We Have Made

We have built a series of interactive web applications that allow students to learn Artificial Intelligence without installing any software. Each torchjs##.html file corresponds approximately to a tinyTorch Python step, but adds visual feedback that only a browser can easily provide:

Live Visuals: Students can see their camera feed and the "Binary View" (what the AI sees) side-by-side.

Instant Interaction: Buttons allow for manual data fetching, real-time training, and immediate weight exporting.

No-Setup Training: By using the computer's GPU through the browser, students can train models on their own data in seconds. <br>

## How to Use It

Open the File: Simply double-click any .html file to open it in Chrome, Edge, or any modern browser.

Start the Hardware: Click the "Start Camera" button to grant the browser permission to see your webcam.

Load or Train: Depending on the file, use the buttons to either fetch professional data from the internet or capture your own drawings.

Click the black textarea to see running code, copy it for your own work, change it, click the "Update and Run" button to live see your changes. 

Export: Once the AI is performing well, use the "Export .h" button to save the "brain" as a C++ header file, ready to be flashed onto an ESP32-S3 microcontroller. (This part is still in draft mode)



[**Big Picture Overview**](https://mlsysbook.ai/tinytorch/big-picture.html)




## 🚀 The Full Curriculum (Math to Microcontroller)

| File | Title | Concept Learned | tinyTorch Chapter |
| :--- | :--- | :--- | :--- |
| [**torchjs01.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs01.html) | Scalar Tensors | Creating basic numbers and the "Tensor" data type. | [**1.1** The Tensor Class](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html#constructor) |
| [**torchjs02.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs02.html) | Autograd Math | How TFjs tracks math history for derivatives. | [**6.1** Differentiation](https://mlsysbook.ai/tinytorch/modules/06_autograd_ABOUT.html) |
| [**torchjs03.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs03.html) | Linear Regression | Solving the y = Mx + B line equation. | [**2.1** Linear Regression](https://mlsysbook.ai/tinytorch/modules/02_activations_ABOUT.html#core-concepts) |
| [**torchjs04.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs04.html) | The First Neuron | Building a single-layer neural network. | [**2.2** Neurons & Perceptrons](https://mlsysbook.ai/tinytorch/modules/02_activations_ABOUT.html#) |
| [**torchjs05.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs05.html) | Hidden Layers | Adding "inner thoughts" for non-linear problems. | [**3.1** Multi-Layer Perceptrons](https://mlsysbook.ai/tinytorch/modules/03_layers_ABOUT.html) |
| [**torchjs06.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs06.html) | Optimizers | Using SGD and Adam to "walk" toward the answer. | [**7.1** Optimization](https://mlsysbook.ai/tinytorch/modules/07_optimizers_ABOUT.html) |
| [**torchjs07.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs07.html) | Camera Intro | Connecting the webcam to the browser canvas. | [**4.1** Data Loading](https://mlsysbook.ai/tinytorch/modules/05_dataloader_ABOUT.html) |
| [**torchjs08.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs08.html) | Motion Detect | Using frame differencing to see movement. | [**4.2** Image Ops](https://mlsysbook.ai/tinytorch/modules/04_losses_ABOUT.html) |
| [**torchjs09.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs09.html) | Binary Classifier | Training the brain to see "Hand" vs "Background". | [**2.2** Binary Classification](https://mlsysbook.ai/tinytorch/modules/02_activations_ABOUT.html) |
| [**torchjs10.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs10.html) | Persistent Memory | Saving and Loading weights to local files. | [**6.1** Serialization](https://mlsysbook.ai/tinytorch/modules/06_autograd_ABOUT.html) |
| [**torchjs11.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs11.html) | Multiclass Logic | Expanding to 3+ classes (Hand, Object, Empty). | [**5.5** Softmax & Multiclass](https://mlsysbook.ai/tinytorch/modules/02_activations_ABOUT.html#softmax-and-numerical-stability) |
| [**torchjs12.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs12.html) | Data Augment | Cropping and resizing images to make AI smarter. | [**4.3** Augmentation](https://mlsysbook.ai/tinytorch/modules/04_losses_ABOUT.html) |
| [**torchjs13.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs13.html) | Validation Logic | Using 20% of data to "test" the brain's honesty. | [**7.1** Evaluation](https://mlsysbook.ai/tinytorch/modules/07_optimizers_ABOUT.html) |
| [**torchjs14.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs14.html) | Feature Maps | Visualizing the Conv1 filters (The Brain's Eyes). | [**8.1** Visualization](https://mlsysbook.ai/tinytorch/modules/08_training_ABOUT.html) |
| [**torchjs15.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs15.html) | Export Master | Converting weights to C++ Header (.h) arrays. | [**9.1** Deployment Bridge](https://mlsysbook.ai/tinytorch/modules/09_convolutions_ABOUT.html) |
| [**torchjs16.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs16.html) | MNIST Trainer | Training on 60,000 professional digit samples. | [**10.1** Standard Datasets](https://mlsysbook.ai/tinytorch/modules/09_convolutions_ABOUT.html) |
| [**torchjs17.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs17.html) | Live Digit Reader | Live webcam OCR using Adaptive Thresholding. | [**10.2** Real-world Inference](https://mlsysbook.ai/tinytorch/modules/09_convolutions_ABOUT.html) |
[**torchjs18.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs18.html) | Tokenization | Turning words into numbers (Vocabulary IDs). | [**10.1** Tokenization](https://mlsysbook.ai/tinytorch/modules/10_tokenization_ABOUT.html#) |
| [**torchjs19.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs19.html) | Word Embeddings | Vector math: Moving words into a semantic map. | [**11.2** Embeddings](https://mlsysbook.ai/tinytorch/modules/11_embeddings_ABOUT.html) |
| [**torchjs20.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs20.html) | Self-Attention | The "Focus" mechanism: How words relate to context. | [**12.1** Attention](https://mlsysbook.ai/tinytorch/modules/12_attention_ABOUT.html) |
| [**torchjs21.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs21.html) | Probabilistic AI | Predicting the next word using Softmax weights. | [**11.1** Embeddings](https://mlsysbook.ai/tinytorch/modules/11_embeddings_ABOUT.html) |
| [**torchjs22.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs22.html) | Quantization | Shrinking bits (32-bit to 8-bit) for tiny chips. | [**15.1** Quantization](https://mlsysbook.ai/tinytorch/modules/15_quantization_ABOUT.html) |
| [**torchjs23.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs23.html) | Brain Pruning | Deleting "lazy" neurons to speed up the network. | [**16.1** Pruning Ops](https://mlsysbook.ai/tinytorch/modules/16_compression_ABOUT.html) |
| [**torchjs24.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs24.html) | Performance Lab | Benchmarking Latency and FPS in real-time. | [**19.1** Benchmarking](https://mlsysbook.ai/tinytorch/modules/19_benchmarking_ABOUT.html) |
| [**torchjs25.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs25.html) | The Memory Guard | Debugging GPU leaks with `.dispose()` logic. | [**1.1** Memory Management](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html) |
 [**torchjs00.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs00.html) | Finale JS to XIAOML Kit | Putting Vision all together for the [XIAO ML Kit (esp32S3)](https://www.seeedstudio.com/The-XIAOML-Kit.html). (DRAFT note larger numbers are the most recent models) | [**1.1** Memory Management](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html) |
 [**esp-on-device-train 3 programs**](https://github.com/hpssjellis/my-examples-of-tensorflowjs-for-tinytorch/tree/main/esp-on-device-train) | Got full on-device training working using 3 programs: record images, train, inference. I like this as the steps are clear, more work moving images into folders. Folder names determine classes, max 30 images each class | Putting Vision all together on the [XIAO ML Kit (esp32S3)](https://www.seeedstudio.com/The-XIAOML-Kit.html).  | [**1.1** Memory Management](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html) |
 [**Best esp-on-device-train Single Program**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/esp-on-device-train-one-program/esp-all-menu-A0-image-train-infer.txt) | Put the above 3 programs into one with an OLED A0 touch trigger (short and long press) Menu: record 3 classes of images, train, inference. This is very impressive, but a bit buggy at the moment. Does everything without needing to manually move images around on the micro sd card. It works, but don't expect good results yet, that is why we use EdgeImpulse. | Putting Vision all together on the [XIAO ML Kit (esp32S3)](https://www.seeedstudio.com/The-XIAOML-Kit.html).  | [**1.1** Memory Management](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html) |




<h1> Note: Recent stable web final version bare metal vision to the XIAOML kit</h1> 

[torchjs00.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs00.html) is copyied from [torchjs72.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs72.html)

[**Best esp-on-device-train Single Program**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/esp-on-device-train-one-program/esp-all-menu-A0-image-train-infer.txt) esp-all-menu-A0-image-train-infer22.txt as of Jan 15, 2026
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>


### Presently working on: and not fully tested.



Issues: [torchjs75gpt.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs75gpt.html)





Issues: [torchjs76claude.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs76claude.html) 

Issues: [torchjs78claude.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs78claude.html) 

## What is interesting
(The bug is the generated code from the webpage is not working on the esp32s3 and I want the resolution of both cameras to be user decided. Default 64x64 RGB)

Interesting: [torchjs78gemini.html](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs78gemini.html)



Super Interesting is the latest single file code for all image loading training and inference. The trick here is that the images are fairly small 64 x 64 is the default. [esp-on-device-train-one-program/esp-all-menu-A0-image-train-infer21.txt](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/esp-on-device-train-one-program/esp-all-menu-A0-image-train-infer21.txt)

It's a process!

<h2>By Jeremy Ellis, Use at your own Risk</h2>
<a href="https://github.com/hpssjellis">github Profile hpssjellis</a><br>
<a href="https://www.linkedin.com/in/jeremy-ellis-4237a9bb/">LinkedIn jeremy-ellis-4237a9bb</a> <br>
<a href="https://opencollective.com/mlsysbook">Support the opencollective.com/mlsysbook</a> <br>
<a href="https://github.com/hpssjellis/my-examples-of-tensorflowjs-for-tinytorch">This Github is at:my-examples-of-tensorflowjs-for-tinytorch </a> <br>
<a href="https://www.seeedstudio.com/The-XIAOML-Kit.html">The $22 USD xiaoMLkit ($38 USD if you need a usbC cable and a micro SD Card)</a> <br>




## For training   
You can print it














<img width="233" height="245" alt="image-blank-circle-square" src="https://github.com/user-attachments/assets/bf911406-6061-475e-b30a-925f37373f99" />






