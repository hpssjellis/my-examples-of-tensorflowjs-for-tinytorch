# my-examples-of-tensorflowjs-for-tinytorch
as above

This page as a web page index.

https://hpssjellis.github.io/my-examples-of-tensorflowjs-for-tinytorch/




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






## 🚀 The Full Curriculum (Browser-Based AI)

| File | Title | Concept Learned | tinyTorch Chapter |
| :--- | :--- | :--- | :--- |
| **torchjs01** | Tensor Basics | Creating tensors in JavaScript and managing GPU memory. | **1.1** The Tensor Class |
| **torchjs02** | Automatic Gradients | Using `tf.variable` and `tf.tidy` for derivatives. | **1.2** Automatic Differentiation |
| **torchjs03** | Linear Regression | Solving $y = mx + b$ inside a `<canvas>`. | **2.1** Linear Regression |
| **torchjs04** | The Single Neuron | Building a `tf.sequential` model with one layer. | **2.2** Neurons & Perceptrons |
| **torchjs05** | Hidden Thoughts | Deep Networks with ReLU activation in the browser. | **3.1** Multi-Layer Perceptrons |
| **torchjs06** | Browser Optimizers | Watching the Adam optimizer find answers in real-time. | **3.2** Optimization Algorithms |
| **torchjs07** | Camera Streaming | Connecting `getUserMedia` to a video tag and canvas. | **4.1** Data Loading |
| **torchjs08** | Frame Differencing | Basic Computer Vision: Finding motion between frames. | **4.2** Basic Image Ops |
| **torchjs09** | Live Binary Classifier | Training the brain on "Hand" vs "No Hand" via webcam. | **5.1** Binary Classification |
| **torchjs10** | Local Storage | Saving and Loading model weights to the Downloads folder. | **6.1** Model Serialization |
| **torchjs11** | Softmax 3-Class | Teaching the AI to distinguish between three unique items. | **5.2** Softmax & Entropy |
| **torchjs12** | Tensor Manipulation | Cropping and resizing camera pixels for the brain. | **4.3** Augmentation |
| **torchjs13** | Validation Exam | Creating an 80/20 data split for automatic testing. | **7.1** Evaluation |
| **torchjs14** | X-Ray Filter View | Visualizing Convolutional Feature Maps on a grid. | **8.1** Visualization |
| **torchjs15** | C Header Exporter | Generating `.h` files for ESP32-S3 Arduino deployment. | **9.1** Deployment Bridge |
| **torchjs16** | MNIST Data Fetcher | Manual internet fetching of 60,000 professional digits. | **10.1** Standard Datasets |
| **torchjs17** | Live Scanner & Log | Real-time OCR with Adaptive Thresholding and history log. | **10.2** Real-world Inference |
