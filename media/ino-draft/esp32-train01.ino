// esp32-train.ino
// ESP32-S3 on-device training sketch (FLOAT32 training, INT8 or FLOAT export)
// Target: Seeed XIAO ESP32S3 Sense (XIAOML Kit)

#include <Arduino.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include "esp_camera.h"

// ================= USER CONFIG =================
#define MY_IMAGE_W 64
#define MY_IMAGE_H 64
#define MY_NUM_CLASSES 3
#define MY_BATCH_SIZE 2           // small on purpose
#define MY_LEARNING_RATE 0.001f
#define MY_USE_GRAYSCALE           // comment out for RGB
#define MY_EXPORT_INT8             // comment out for float export

#define MY_TRAIN_TRIGGER_PIN A0

// ================= MODEL STORAGE =================
#define MY_MODEL_FOLDER "/models"
#define MY_MODEL_FILENAME "/models/myModel_trained.h"

// ================= MEMORY HELPERS =================
#define MY_PSRAM_ALLOC(size) heap_caps_malloc(size, MALLOC_CAP_SPIRAM)
#define MY_PSRAM_FREE(ptr)   if(ptr){heap_caps_free(ptr); ptr=nullptr;}

// ================= SIMPLE UTIL =================
inline float myLeakyRelu(float x) { return (x > 0) ? x : 0.1f * x; }

// ================= DATA STRUCTURES =================
struct myImageSample {
  float *myData;   // [H*W*C]
  uint8_t myLabel;
};

// ================= MODEL PARAMETERS =================
// Conv1: 3x3, inC -> 4
float *myConv1_w;   // [4][inC][3][3]
float *myConv1_b;   // [4]

// Conv2: 3x3, 4 -> 8
float *myConv2_w;   // [8][4][3][3]
float *myConv2_b;   // [8]

// Output: dense
float *myOut_w;     // [flatten][classes]
float *myOut_b;     // [classes]

// ================= FORWARD BUFFERS =================
float *myConv1_out;
float *myPool1_out;
float *myConv2_out;
float *myFlat_out;
float mySoftmax[MY_NUM_CLASSES];

// ================= TRAINING STATE =================
bool myTrainingRequested = false;
uint32_t myBatchCounter = 0;

// ================= FILESYSTEM =================
bool myInitSD() {
  if (!SD.begin()) {
    Serial.println("SD init failed");
    return false;
  }
  if (!SD.exists(MY_MODEL_FOLDER)) {
    SD.mkdir(MY_MODEL_FOLDER);
  }
  return true;
}

// ================= MODEL INIT =================
void myAllocateModel() {
  const int myInC =
#ifdef MY_USE_GRAYSCALE
    1;
#else
    3;
#endif

  myConv1_w = (float*)MY_PSRAM_ALLOC(4 * myInC * 3 * 3 * sizeof(float));
  myConv1_b = (float*)MY_PSRAM_ALLOC(4 * sizeof(float));
  myConv2_w = (float*)MY_PSRAM_ALLOC(8 * 4 * 3 * 3 * sizeof(float));
  myConv2_b = (float*)MY_PSRAM_ALLOC(8 * sizeof(float));

  int myFlatSize = (MY_IMAGE_W/2 - 2) * (MY_IMAGE_H/2 - 2) * 8;
  myOut_w = (float*)MY_PSRAM_ALLOC(myFlatSize * MY_NUM_CLASSES * sizeof(float));
  myOut_b = (float*)MY_PSRAM_ALLOC(MY_NUM_CLASSES * sizeof(float));

  // Forward buffers
  myConv1_out = (float*)MY_PSRAM_ALLOC(MY_IMAGE_W * MY_IMAGE_H * 4 * sizeof(float));
  myPool1_out = (float*)MY_PSRAM_ALLOC((MY_IMAGE_W/2) * (MY_IMAGE_H/2) * 4 * sizeof(float));
  myConv2_out = (float*)MY_PSRAM_ALLOC((MY_IMAGE_W/2 - 2) * (MY_IMAGE_H/2 - 2) * 8 * sizeof(float));
  myFlat_out  = (float*)MY_PSRAM_ALLOC(myFlatSize * sizeof(float));
}

// ================= WEIGHT INIT =================
void myInitWeights() {
  for (int i = 0; i < 4; i++) myConv1_b[i] = 0.0f;
  for (int i = 0; i < 8; i++) myConv2_b[i] = 0.0f;
  for (int i = 0; i < MY_NUM_CLASSES; i++) myOut_b[i] = 0.0f;

  int myTotal;
  myTotal = 4 * 3 * 3 * 3;
  for (int i = 0; i < myTotal; i++) myConv1_w[i] = random(-100,100)/500.0f;

  myTotal = 8 * 4 * 3 * 3;
  for (int i = 0; i < myTotal; i++) myConv2_w[i] = random(-100,100)/500.0f;
}

// ================= SOFTMAX =================
void mySoftmaxFn(float *myIn, float *myOut) {
  float myMax = myIn[0];
  for (int i=1;i<MY_NUM_CLASSES;i++) if(myIn[i]>myMax) myMax=myIn[i];
  float mySum=0;
  for (int i=0;i<MY_NUM_CLASSES;i++) { myOut[i]=exp(myIn[i]-myMax); mySum+=myOut[i]; }
  for (int i=0;i<MY_NUM_CLASSES;i++) myOut[i]/=mySum;
}

// ================= TRAIN STEP (SINGLE SAMPLE) =================
float myTrainSample(myImageSample &mySample) {
  // NOTE: forward + backward intentionally simplified here
  // Full convolution math omitted for brevity in this first attempt

  // Fake logits for now
  float myLogits[MY_NUM_CLASSES] = {0};
  mySoftmaxFn(myLogits, mySoftmax);

  float myLoss = -log(mySoftmax[mySample.myLabel] + 1e-6f);
  return myLoss;
}

// ================= TRAIN BATCH =================
void myTrainBatch() {
  myBatchCounter++;
  float myLossSum = 0;

  for (int i = 0; i < MY_BATCH_SIZE; i++) {
    // placeholder sample loading
    myImageSample mySample;
    mySample.myLabel = i % MY_NUM_CLASSES;
    myLossSum += myTrainSample(mySample);
  }

  Serial.print("Batch "); Serial.print(myBatchCounter);
  Serial.print(" Loss: "); Serial.println(myLossSum / MY_BATCH_SIZE, 6);
}

// ================= EXPORT HEADER =================
void myExportHeader() {
  File myFile = SD.open(MY_MODEL_FILENAME, FILE_WRITE);
  if (!myFile) {
    Serial.println("Failed to open model file for write");
    return;
  }

  myFile.println("#ifndef MY_MODEL_H");
  myFile.println("#define MY_MODEL_H\n");

#ifdef MY_EXPORT_INT8
  myFile.println("#define USE_INT8_MODE");
#endif
#ifdef MY_USE_GRAYSCALE
  myFile.println("#define USE_GRAYSCALE_MODE\n");
#endif

  myFile.println("// Weights omitted here for brevity");
  myFile.println("#endif");

  myFile.close();
  Serial.println("Model header saved to SD");
}

// ================= SETUP / LOOP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(MY_TRAIN_TRIGGER_PIN, INPUT_PULLUP);

  if (!psramFound()) {
    Serial.println("PSRAM not found!");
    while (1);
  }

  myInitSD();
  myAllocateModel();
  myInitWeights();

  Serial.println("esp32-train ready");
}

void loop() {
  if (digitalRead(MY_TRAIN_TRIGGER_PIN) == LOW) {
    delay(50);
    if (!myTrainingRequested) {
      myTrainingRequested = true;
      Serial.println("Training triggered");
    }
  }

  if (myTrainingRequested) {
    myTrainBatch();
    if (myBatchCounter >= 10) {
      myExportHeader();
      myTrainingRequested = false;
      myBatchCounter = 0;
      Serial.println("Training session complete");
    }
    delay(10);
  }
}
