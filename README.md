# my-examples-of-tensorflowjs-for-tinytorch


This page as a web page index.

https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/




<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/2247019e-524d-4cc0-8087-b717466cebd3" />


![Uploading image.png…]()




What We Have Made
We have built a series of interactive web applications that allow students to learn Artificial Intelligence without installing any software. Each torchjs##.html file corresponds to a Python step, but adds visual feedback that only a browser can provide:

Live Visuals: Students can see their camera feed and the "Binary View" (what the AI sees) side-by-side.

Instant Interaction: Buttons allow for manual data fetching, real-time training, and immediate weight exporting.

No-Setup Training: By using the computer's GPU through the browser, students can train models on their own handwriting in seconds.

How to Use It
Open the File: Simply double-click any .html file to open it in Chrome, Edge, or any modern browser.

Start the Hardware: Click the "Start Camera" button to grant the browser permission to see your webcam.

Load or Train: Depending on the file, use the buttons to either fetch professional data from the internet or capture your own drawings.

Export: Once the AI is performing well, use the "Export .h" button to save the "brain" as a C++ header file, ready to be flashed onto an ESP32-S3 microcontroller.






## 🚀 The Full Curriculum (Math to Microcontroller)

| File | Title | Concept Learned | tinyTorch Chapter |
| :--- | :--- | :--- | :--- |
| [**torchjs01.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs01.html) | Scalar Tensors | Creating basic numbers and the "Tensor" data type. | [**1.1** The Tensor Class](https://mlsysbook.ai/tinytorch/modules/01_tensor_ABOUT.html#constructor) |
| [**torchjs02.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs02.html) | Autograd Math | How TFjs tracks math history for derivatives. | [**1.2** Differentiation](https://mlsysbook.ai/tinytorch/modules/06_autograd_ABOUT.html) |
| [**torchjs03.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs03.html) | Linear Regression | Solving the $y = mx + b$ line equation. | [**2.1** Linear Regression](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs04.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs04.html) | The First Neuron | Building a single-layer neural network. | [**2.2** Neurons & Perceptrons](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs05.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs05.html) | Hidden Layers | Adding "inner thoughts" for non-linear problems. | [**3.1** Multi-Layer Perceptrons](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs06.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs06.html) | Optimizers | Using SGD and Adam to "walk" toward the answer. | [**3.2** Optimization](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs07.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs07.html) | Camera Intro | Connecting the webcam to the browser canvas. | [**4.1** Data Loading](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs08.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs08.html) | Motion Detect | Using frame differencing to see movement. | [**4.2** Image Ops](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs09.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs09.html) | Binary Classifier | Training the brain to see "Hand" vs "Background". | [**5.1** Binary Classification](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs10.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs10.html) | Persistent Memory | Saving and Loading weights to local files. | [**6.1** Serialization](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs11.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs11.html) | Multiclass Logic | Expanding to 3+ classes (Hand, Object, Empty). | [**5.2** Softmax & Entropy](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs12.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs12.html) | Data Augment | Cropping and resizing images to make AI smarter. | [**4.3** Augmentation](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs13.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs13.html) | Validation Logic | Using 20% of data to "test" the brain's honesty. | [**7.1** Evaluation](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs14.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs14.html) | Feature Maps | Visualizing the Conv1 filters (The Brain's Eyes). | [**8.1** Visualization](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs15.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs15.html) | Export Master | Converting weights to C++ Header (.h) arrays. | [**9.1** Deployment Bridge](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs16.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs16.html) | MNIST Trainer | Training on 60,000 professional digit samples. | [**10.1** Standard Datasets](https://mlsysbook.ai/tinytorch/intro.html) |
| [**torchjs17.html**](https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/torchjs17.html) | Live Digit Reader | Live webcam OCR using Adaptive Thresholding. | [**10.2** Real-world Inference](https://mlsysbook.ai/tinytorch/intro.html) |

[**Big Picture Overview**](https://mlsysbook.ai/tinytorch/big-picture.html)




