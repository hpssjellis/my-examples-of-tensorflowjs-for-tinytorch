/*
 * ESP32-S3 On-Device CNN Training
 * Images stored in /images/[label]/
 * Weights saved/loaded from /header/myModel.h
 */

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>

// USER PARAMETERS
#define USE_GRAYSCALE_MODE false
#define USE_INT8_QUANTIZATION false
float myLearningRate = 0.001;
float myDropoutRate = 0.3;
int myBatchSize = 6;
int myTargetEpochs = 10;
int myMaxImagesPerClass = 100;
bool myUseAugmentation = true;
float myBrightnessRange = 0.2;
float myContrastRange = 0.4;
bool myInitializedWeights = false;

// Persistence flag
bool myInitialWeightsLoaded = false;

// CAMERA PINS
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

// MODEL ARCHITECTURE
const int myInputChannels = USE_GRAYSCALE_MODE ? 1 : 3;
const int myFlattenedSize = 6728;

// WEIGHTS
float *myConv1_w, *myConv1_b, *myConv2_w, *myConv2_b, *myDense_w, *myDense_b;
float *myConv1_w_grad, *myConv1_b_grad, *myConv2_w_grad, *myConv2_b_grad, *myDense_w_grad, *myDense_b_grad;
float *myConv1_w_m, *myConv1_w_v, *myConv1_b_m, *myConv1_b_v;
float *myConv2_w_m, *myConv2_w_v, *myConv2_b_m, *myConv2_b_v;
float *myDense_w_m, *myDense_w_v, *myDense_b_m, *myDense_b_v;

// BUFFERS
float *myInputBuffer, *myConv1Output, *myPool1Output, *myConv2Output, *myDropoutMask, *myDenseOutput;
float *myDenseGrad, *myConv2Grad, *myPool1Grad, *myConv1Grad;

// DATA
struct MyTrainingImage { float* data; int label; };
std::vector<MyTrainingImage> myTrainingData;
String myClassLabels[3];
int myClassCounts[3] = {0,0,0};

// UTILITY
inline float myClipValue(float v, float mn=-100, float mx=100) {
  if(isnan(v)||isinf(v)) return 0;
  return constrain(v,mn,mx);
}
inline float myLeakyRelu(float x) { return x>0 ? x : 0.1f*x; }
inline float myLeakyReluDeriv(float x) { return x>0 ? 1.0f : 0.1f; }
float myRandomFloat(float mn, float mx) { return mn + (float)random(10000)/10000.0f*(mx-mn); }

String myToCString(String s) {
  s = s.substring(0,20);
  s.replace("\\","\\\\"); s.replace("\"","\\\""); s.replace("\n","\\n");
  return s;
}

// Helper to parse weights for persistence
void myParseWeightLine(File &myFile, float* myTargetBuffer, int mySize) {
    if (!myFile.find("{")) return;
    for (int i = 0; i < mySize; i++) {
        myTargetBuffer[i] = myFile.parseFloat();
    }
}

// MEMORY ALLOCATION
bool myAllocateModelMemory() {
  Serial.println("\n=== Allocating Memory ===");
  int c1w_sz = 3*3*myInputChannels*4;
  int c2w_sz = 3*3*4*8;
  int dw_sz = myFlattenedSize*3;
  
  myConv1_w = (float*)ps_malloc(c1w_sz*sizeof(float));
  myConv1_b = (float*)ps_malloc(4*sizeof(float));
  myConv2_w = (float*)ps_malloc(c2w_sz*sizeof(float));
  myConv2_b = (float*)ps_malloc(8*sizeof(float));
  myDense_w = (float*)ps_malloc(dw_sz*sizeof(float));
  myDense_b = (float*)ps_malloc(3*sizeof(float));
  
  myConv1_w_grad = (float*)ps_malloc(c1w_sz*sizeof(float));
  myConv1_b_grad = (float*)ps_malloc(4*sizeof(float));
  myConv2_w_grad = (float*)ps_malloc(c2w_sz*sizeof(float));
  myConv2_b_grad = (float*)ps_malloc(8*sizeof(float));
  myDense_w_grad = (float*)ps_malloc(dw_sz*sizeof(float));
  myDense_b_grad = (float*)ps_malloc(3*sizeof(float));
  
  myConv1_w_m = (float*)ps_calloc(c1w_sz, sizeof(float));
  myConv1_w_v = (float*)ps_calloc(c1w_sz, sizeof(float));
  myConv1_b_m = (float*)ps_calloc(4, sizeof(float));
  myConv1_b_v = (float*)ps_calloc(4, sizeof(float));
  myConv2_w_m = (float*)ps_calloc(c2w_sz, sizeof(float));
  myConv2_w_v = (float*)ps_calloc(c2w_sz, sizeof(float));
  myConv2_b_m = (float*)ps_calloc(8, sizeof(float));
  myConv2_b_v = (float*)ps_calloc(8, sizeof(float));
  myDense_w_m = (float*)ps_calloc(dw_sz, sizeof(float));
  myDense_w_v = (float*)ps_calloc(dw_sz, sizeof(float));
  myDense_b_m = (float*)ps_calloc(3, sizeof(float));
  myDense_b_v = (float*)ps_calloc(3, sizeof(float));
  
  myInputBuffer = (float*)ps_malloc(64*64*myInputChannels*sizeof(float));
  myConv1Output = (float*)ps_malloc(62*62*4*sizeof(float));
  myPool1Output = (float*)ps_malloc(31*31*4*sizeof(float));
  myConv2Output = (float*)ps_malloc(29*29*8*sizeof(float));
  myDropoutMask = (float*)ps_malloc(myFlattenedSize*sizeof(float));
  myDenseOutput = (float*)ps_malloc(3*sizeof(float));
  
  myDenseGrad = (float*)ps_malloc(myFlattenedSize*sizeof(float));
  myConv2Grad = (float*)ps_malloc(29*29*8*sizeof(float));
  myPool1Grad = (float*)ps_malloc(31*31*4*sizeof(float));
  myConv1Grad = (float*)ps_malloc(62*62*4*sizeof(float));
  
  if(!myConv1_w || !myConv2_w || !myDense_w) { Serial.println("[ERR] Alloc failed!"); return false; }
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  return true;
}

void myInitializeWeights() {
  if (myInitializedWeights) return;

  // Try to load existing header for persistence
  File myHeaderFile = SD.open("/header/myModel.h", FILE_READ);
  if (myHeaderFile) {
      Serial.println("Existing model found. Loading weights...");
      int c1w_sz = USE_GRAYSCALE_MODE ? 36 : 108;
      myParseWeightLine(myHeaderFile, myConv1_w, c1w_sz);
      myParseWeightLine(myHeaderFile, myConv1_b, 4);
      myParseWeightLine(myHeaderFile, myConv2_w, 288);
      myParseWeightLine(myHeaderFile, myConv2_b, 8);
      myParseWeightLine(myHeaderFile, myDense_w, myFlattenedSize * 3);
      myParseWeightLine(myHeaderFile, myDense_b, 3);
      myHeaderFile.close();
      myInitializedWeights = true;
      myInitialWeightsLoaded = true;
      return;
  }

  Serial.println("Initializing with random weights...");
  int c1w_sz = USE_GRAYSCALE_MODE ? 36 : 108;
  float c1std = sqrt(2.0/(9.0*myInputChannels));
  for(int i=0; i<c1w_sz; i++) myConv1_w[i] = myRandomFloat(-c1std, c1std);
  for(int i=0; i<4; i++) myConv1_b[i] = 0;
  float c2std = sqrt(2.0/36.0);
  for(int i=0; i<288; i++) myConv2_w[i] = myRandomFloat(-c2std, c2std);
  for(int i=0; i<8; i++) myConv2_b[i] = 0;
  float dstd = sqrt(2.0/myFlattenedSize);
  for(int i=0; i<myFlattenedSize*3; i++) myDense_w[i] = myRandomFloat(-dstd, dstd);
  for(int i=0; i<3; i++) myDense_b[i] = 0;
  
  myInitializedWeights = true;
}

// IMAGE LOADING
bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if(!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if(!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  uint8_t* rgb = (uint8_t*)ps_malloc(320*240*3);
  if(!rgb) { free(jpg); return false; }
  
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, rgb);
  free(jpg);
  if(!ok) { free(rgb); return false; }
  
  for(int y=0; y<64; y++) {
    for(int x=0; x<64; x++) {
      int sy = (int)((y+0.5)*240.0/64.0);
      int sx = (int)((x+0.5)*320.0/64.0);
      if(sy>239) sy=239;
      if(sx>319) sx=319;
      int idx = (sy*320+sx)*3;
      if(USE_GRAYSCALE_MODE) {
        buf[y*64+x] = (rgb[idx]*0.299 + rgb[idx+1]*0.587 + rgb[idx+2]*0.114)/255.0;
      } else {
        int b = (y*64+x)*3;
        buf[b] = rgb[idx]/255.0;
        buf[b+1] = rgb[idx+1]/255.0;
        buf[b+2] = rgb[idx+2]/255.0;
      }
    }
  }
  
  free(rgb);
  return true;
}

void myLoadImagesFromSd() {
  Serial.println("\n=== Loading Images ===");
  if (!SD.exists("/images")) SD.mkdir("/images");
  File root = SD.open("/images");
  if(!root) { Serial.println("[ERR] /images folder missing"); return; }
  
  int cls = 0;
  File folder = root.openNextFile();
  while(folder && cls < 3) {
    if(folder.isDirectory()) {
      String name = String(folder.name());
      if(name.startsWith(".") || name=="header") {
        folder = root.openNextFile();
        continue;
      }
      
      myClassLabels[cls] = name;
      Serial.printf("\nClass %d: %s\n", cls, name.c_str());
      File img = folder.openNextFile();
      int loaded = 0;
      
      while(img && loaded < myMaxImagesPerClass) {
        if(!img.isDirectory()) {
          String fn = String(img.name());
          if(fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
            String path = "/images/" + name + "/" + fn;
            float* ib = (float*)ps_malloc(64*64*myInputChannels*sizeof(float));
            if(ib && myLoadImageFromFile(path.c_str(), ib)) {
              MyTrainingImage ti;
              ti.data = ib;
              ti.label = cls;
              myTrainingData.push_back(ti);
              loaded++;
              myClassCounts[cls]++;
            } else {
              if(ib) free(ib);
            }
          }
        }
        img = folder.openNextFile();
      }
      
      Serial.printf("  Loaded: %d\n", loaded);
      cls++;
    }
    folder = root.openNextFile();
  }
  
  Serial.printf("\nTotal: %d [%d,%d,%d]\n", myTrainingData.size(), myClassCounts[0], myClassCounts[1], myClassCounts[2]);
}

void myAugmentImage(float* src, float* dst) {
  int sz = 64*64*myInputChannels;
  if(!myUseAugmentation) { memcpy(dst, src, sz*sizeof(float)); return; }
  
  float br = myRandomFloat(-myBrightnessRange, myBrightnessRange);
  float co = myRandomFloat(1.0-myContrastRange/2, 1.0+myContrastRange/2);
  for(int i=0; i<sz; i++) {
    dst[i] = myClipValue((src[i]-0.5)*co+0.5+br, 0.0, 1.0);
  }
}

// FORWARD PASS FUNCTIONS
void myForwardConv1() {
  for(int f=0; f<4; f++) {
    int ob = f*3844;
    for(int y=0; y<62; y++) {
      for(int x=0; x<62; x++) {
        float s = 0;
        if(USE_GRAYSCALE_MODE) {
          for(int ky=0; ky<3; ky++)
            for(int kx=0; kx<3; kx++)
              s += myInputBuffer[(y+ky)*64+(x+kx)] * myConv1_w[f*9+ky*3+kx];
        } else {
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int p = ((y+ky)*64+(x+kx))*3;
              int w = f*27+ky*9+kx*3;
              s += myInputBuffer[p]*myConv1_w[w] + myInputBuffer[p+1]*myConv1_w[w+1] + myInputBuffer[p+2]*myConv1_w[w+2];
            }
          }
        }
        myConv1Output[ob+y*62+x] = myLeakyRelu(myClipValue(s+myConv1_b[f]));
      }
    }
  }
}

void myForwardPool1() {
  for(int f=0; f<4; f++) {
    int ib=f*3844, ob=f*961;
    for(int y=0; y<31; y++) {
      for(int x=0; x<31; x++) {
        int iy=y*2, ix=x*2;
        float m = myConv1Output[ib+iy*62+ix];
        m = max(m, myConv1Output[ib+iy*62+ix+1]);
        m = max(m, myConv1Output[ib+(iy+1)*62+ix]);
        m = max(m, myConv1Output[ib+(iy+1)*62+ix+1]);
        myPool1Output[ob+y*31+x] = m;
      }
    }
  }
}

void myForwardConv2() {
  for(int f=0; f<8; f++) {
    int ob=f*841;
    for(int y=0; y<29; y++) {
      for(int x=0; x<29; x++) {
        float s = 0;
        for(int c=0; c<4; c++) {
          int ib=c*961;
          for(int ky=0; ky<3; ky++)
            for(int kx=0; kx<3; kx++)
              s += myPool1Output[ib+(y+ky)*31+(x+kx)] * myConv2_w[f*36+c*9+ky*3+kx];
        }
        myConv2Output[ob+y*29+x] = myLeakyRelu(myClipValue(s+myConv2_b[f]));
      }
    }
  }
}

void myForwardDropout(bool training) {
  if(training && myDropoutRate>0) {
    float kp = 1.0-myDropoutRate;
    for(int i=0; i<myFlattenedSize; i++) {
      myDropoutMask[i] = (myRandomFloat(0,1)<kp) ? (1.0/kp) : 0.0;
      myConv2Output[i] *= myDropoutMask[i];
    }
  } else {
    for(int i=0; i<myFlattenedSize; i++) myDropoutMask[i] = 1.0;
  }
}

void myForwardDense() {
  for(int c=0; c<3; c++) {
    double s=0, comp=0;
    for(int i=0; i<myFlattenedSize; i++) {
      double t = myConv2Output[i]*myDense_w[c*myFlattenedSize+i];
      double y = t-comp;
      double tt = s+y;
      comp = (tt-s)-y;
      s = tt;
    }
    myDenseOutput[c] = myClipValue((float)s+myDense_b[c], -50, 50);
  }
  
  float mx = max(max(myDenseOutput[0], myDenseOutput[1]), myDenseOutput[2]);
  float es = exp(myDenseOutput[0]-mx) + exp(myDenseOutput[1]-mx) + exp(myDenseOutput[2]-mx);
  for(int i=0; i<3; i++) myDenseOutput[i] = exp(myDenseOutput[i]-mx)/es;
}

// BACKWARD PASS FUNCTIONS
void myBackwardDense(int lbl) {
  for(int c=0; c<3; c++) {
    float e = myDenseOutput[c] - (c==lbl ? 1.0f : 0.0f);
    for(int i=0; i<myFlattenedSize; i++) {
      myDense_w_grad[c*myFlattenedSize+i] = e*myConv2Output[i];
      myDenseGrad[i] = (c==0) ? e*myDense_w[c*myFlattenedSize+i] : myDenseGrad[i]+e*myDense_w[c*myFlattenedSize+i];
    }
    myDense_b_grad[c] = e;
  }
}

void myBackwardDropout() {
  for(int i=0; i<myFlattenedSize; i++) myDenseGrad[i] *= myDropoutMask[i];
}

void myBackwardConv2() {
  for(int i=0; i<myFlattenedSize; i++) myConv2Grad[i] = myDenseGrad[i]*myLeakyReluDeriv(myConv2Output[i]);
  
  memset(myConv2_w_grad, 0, 288*sizeof(float));
  memset(myConv2_b_grad, 0, 8*sizeof(float));
  memset(myPool1Grad, 0, 3844*sizeof(float));
  for(int f=0; f<8; f++) {
    int ob=f*841;
    for(int y=0; y<29; y++) {
      for(int x=0; x<29; x++) {
        float g = myConv2Grad[ob+y*29+x];
        myConv2_b_grad[f] += g;
        for(int c=0; c<4; c++) {
          int ib=c*961;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int pi = ib+(y+ky)*31+(x+kx);
              int wi = f*36+c*9+ky*3+kx;
              myConv2_w_grad[wi] += g*myPool1Output[pi];
              myPool1Grad[pi] += g*myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

void myBackwardPool1() {
  memset(myConv1Grad, 0, 15376*sizeof(float));
  for(int f=0; f<4; f++) {
    int ib=f*3844, ob=f*961;
    for(int y=0; y<31; y++) {
      for(int x=0; x<31; x++) {
        int iy=y*2, ix=x*2;
        float pv = myPool1Output[ob+y*31+x];
        float g = myPool1Grad[ob+y*31+x];
        if(myConv1Output[ib+iy*62+ix] == pv) myConv1Grad[ib+iy*62+ix] += g;
        if(myConv1Output[ib+iy*62+ix+1] == pv) myConv1Grad[ib+iy*62+ix+1] += g;
        if(myConv1Output[ib+(iy+1)*62+ix] == pv) myConv1Grad[ib+(iy+1)*62+ix] += g;
        if(myConv1Output[ib+(iy+1)*62+ix+1] == pv) myConv1Grad[ib+(iy+1)*62+ix+1] += g;
      }
    }
  }
}

void myBackwardConv1() {
  for(int i=0; i<15376; i++) myConv1Grad[i] *= myLeakyReluDeriv(myConv1Output[i]);
  int wsz = USE_GRAYSCALE_MODE ? 36 : 108;
  memset(myConv1_w_grad, 0, wsz*sizeof(float));
  memset(myConv1_b_grad, 0, 4*sizeof(float));
  for(int f=0; f<4; f++) {
    int ob=f*3844;
    for(int y=0; y<62; y++) {
      for(int x=0; x<62; x++) {
        float g = myConv1Grad[ob+y*62+x];
        myConv1_b_grad[f] += g;
        
        if(USE_GRAYSCALE_MODE) {
          for(int ky=0; ky<3; ky++)
            for(int kx=0; kx<3; kx++)
              myConv1_w_grad[f*9+ky*3+kx] += g*myInputBuffer[(y+ky)*64+(x+kx)];
        } else {
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int p = ((y+ky)*64+(x+kx))*3;
              int w = f*27+ky*9+kx*3;
              myConv1_w_grad[w] += g*myInputBuffer[p];
              myConv1_w_grad[w+1] += g*myInputBuffer[p+1];
              myConv1_w_grad[w+2] += g*myInputBuffer[p+2];
            }
          }
        }
      }
    }
  }
}

// ADAM UPDATE
void myAdamUpdate(float* w, float* g, float* m, float* v, int sz, int step) {
  float b1=0.9, b2=0.999, eps=1e-8;
  float lr_t = myLearningRate * sqrt(1-pow(b2,step)) / (1-pow(b1,step));
  for(int i=0; i<sz; i++) {
    m[i] = b1*m[i] + (1-b1)*g[i];
    v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
    w[i] -= lr_t*m[i]/(sqrt(v[i])+eps);
    w[i] = myClipValue(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  int c1sz = USE_GRAYSCALE_MODE ? 36 : 108;
  myAdamUpdate(myConv1_w, myConv1_w_grad, myConv1_w_m, myConv1_w_v, c1sz, step);
  myAdamUpdate(myConv1_b, myConv1_b_grad, myConv1_b_m, myConv1_b_v, 4, step);
  myAdamUpdate(myConv2_w, myConv2_w_grad, myConv2_w_m, myConv2_w_v, 288, step);
  myAdamUpdate(myConv2_b, myConv2_b_grad, myConv2_b_m, myConv2_b_v, 8, step);
  myAdamUpdate(myDense_w, myDense_w_grad, myDense_w_m, myDense_w_v, myFlattenedSize*3, step);
  myAdamUpdate(myDense_b, myDense_b_grad, myDense_b_m, myDense_b_v, 3, step);
}

// TRAINING
void myTrainModel() {
  Serial.println("\n======== TRAINING START ========");
  int total = myTrainingData.size();
  int batches_per_epoch = (total + myBatchSize - 1) / myBatchSize;
  int total_batches = myTargetEpochs * batches_per_epoch;
  
  std::vector<int> indices;
  for(int i=0; i<total; i++) indices.push_back(i);
  
  float running_loss = 0;
  int loss_count = 0;
  for(int batch=0; batch<total_batches; batch++) {
    if(batch % batches_per_epoch == 0) {
      Serial.printf("\n--- Epoch %d/%d ---\n", batch/batches_per_epoch + 1, myTargetEpochs);
      for(int i=total-1; i>0; i--) {
        int j = random(i+1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
    }
    
    int batch_start = (batch % batches_per_epoch) * myBatchSize;
    int batch_end = min(batch_start + myBatchSize, total);
    
    float batch_loss = 0;
    int correct = 0;
    for(int i=batch_start; i<batch_end; i++) {
      int idx = indices[i];
      MyTrainingImage& img = myTrainingData[idx];
      
      myAugmentImage(img.data, myInputBuffer);
      myForwardConv1();
      myForwardPool1();
      myForwardConv2();
      myForwardDropout(true);
      myForwardDense();
      
      myBackwardDense(img.label);
      myBackwardDropout();
      myBackwardConv2();
      myBackwardPool1();
      myBackwardConv1();
      
      float loss = 0;
      for(int c=0; c<3; c++) {
        float target = (c == img.label) ? 1.0f : 0.0f;
        loss -= target * log(max(myDenseOutput[c], 1e-7f));
      }
      batch_loss += loss;
      int pred = (myDenseOutput[1]>myDenseOutput[0]) ? 1 : 0;
      if(myDenseOutput[2]>myDenseOutput[pred]) pred = 2;
      if(pred == img.label) correct++;
    }
    
    myUpdateWeights(batch+1);
    
    running_loss += batch_loss / (batch_end - batch_start);
    loss_count++;
    if((batch+1) % 10 == 0) {
      float avg_loss = running_loss / loss_count;
      float acc = 100.0 * correct / (batch_end - batch_start);
      Serial.printf("Batch %d/%d - Loss: %.4f, Acc: %.1f%%\n", batch+1, total_batches, avg_loss, acc);
      running_loss = 0;
      loss_count = 0;
    }
  }
  
  Serial.println("\n======== TRAINING COMPLETE ========");
}

// DEBUG OUTPUT
void myPrintDebug() {
  Serial.println("\n========== DEBUG OUTPUT ==========");
  Serial.printf("Mode: %s %s\n", USE_GRAYSCALE_MODE?"GRAY":"RGB", USE_INT8_QUANTIZATION?"INT8":"FLOAT");
  Serial.printf("Classes: [%s, %s, %s]\n", myClassLabels[0].c_str(), myClassLabels[1].c_str(), myClassLabels[2].c_str());
  
  if(myTrainingData.size() > 0) {
      myAugmentImage(myTrainingData[0].data, myInputBuffer);
      myForwardConv1();
      myForwardPool1();
      myForwardConv2();
      myForwardDropout(false);
      myForwardDense();
      Serial.println("\nLayer Stats:");
      
      auto print_stats = [](const char* name, float* buf, int len) {
        float mn=1e6, mx=-1e6, sum=0;
        int neg=0, zeros=0;
        for(int i=0; i<len; i++) {
          if(buf[i]<mn) mn=buf[i];
          if(buf[i]>mx) mx=buf[i];
          sum += buf[i];
          if(buf[i]<0) neg++;
          if(buf[i]==0) zeros++;
        }
        Serial.printf("  %s: Min=%.2f Max=%.2f Avg=%.2f Neg=%d%% Zeros=%d%%\n", name, mn, mx, sum/len, (neg*100)/len, (zeros*100)/len);
      };
      
      print_stats("CONV1", myConv1Output, 15376);
      print_stats("POOL1", myPool1Output, 3844);
      print_stats("CONV2", myConv2Output, myFlattenedSize);
      
      Serial.println("\nWeight Samples:");
      Serial.print("  Conv1_w[0-5]: ");
      for(int i=0; i<6; i++) { Serial.print(myConv1_w[i],4); Serial.print(" "); }
      Serial.println();
      
      Serial.print("  Conv1_b[0-3]: ");
      for(int i=0; i<4; i++) { Serial.print(myConv1_b[i],4); Serial.print(" "); }
      Serial.println();
      
      Serial.println("\nLogits:");
      for(int i=0; i<3; i++) {
        Serial.printf("  Class %d (%s): %.4f\n", i, myClassLabels[i].c_str(), log(max(myDenseOutput[i], 1e-7f)));
      }
      
      Serial.println("\nProbabilities:");
      Serial.printf("  [%.1f%%, %.1f%%, %.1f%%]\n", myDenseOutput[0]*100, myDenseOutput[1]*100, myDenseOutput[2]*100);
  }
  
  Serial.println("===================================\n");
}

// SAVE MODEL TO SD CARD
bool mySaveModelHeader() {
  Serial.println("\n=== Saving Model ===");
  if(!SD.exists("/header")) SD.mkdir("/header");
  if(SD.exists("/header/myModel.h")) {
    SD.remove("/header/myModel.h");
    delay(100);
  }
  
  File file = SD.open("/header/myModel.h", FILE_WRITE);
  if(!file) return false;
  
  file.println("// Auto-generated model header");
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H\n");
  
  file.println("const char* myClassLabels[3] = {");
  for(int i=0; i<3; i++) {
    file.print("  \""); file.print(myToCString(myClassLabels[i])); file.print("\"");
    if(i<2) file.println(","); else file.println("");
  }
  file.println("};");
  
  int c1w_sz = USE_GRAYSCALE_MODE ? 36 : 108;
  file.print("const float myConv1_w[] = { ");
  for(int i=0; i<c1w_sz; i++) {
    file.print(myConv1_w[i], 6); file.print("f");
    if(i < c1w_sz-1) file.print(",");
  }
  file.println(" };");

  file.print("const float myConv1_b[] = { ");
  for(int i=0; i<4; i++) {
    file.print(myConv1_b[i], 6); file.print("f");
    if(i < 3) file.print(",");
  }
  file.println(" };");
  
  file.print("const float myConv2_w[] = { ");
  for(int i=0; i<288; i++) {
    file.print(myConv2_w[i], 6); file.print("f");
    if(i < 287) file.print(",");
  }
  file.println(" };");

  file.print("const float myConv2_b[] = { ");
  for(int i=0; i<8; i++) {
    file.print(myConv2_b[i], 6); file.print("f");
    if(i < 7) file.print(",");
  }
  file.println(" };");

  file.print("const float myOutput_w[] = { ");
  for(int i=0; i<myFlattenedSize*3; i++) {
    file.print(myDense_w[i], 6); file.print("f");
    if(i < myFlattenedSize*3-1) file.print(",");
  }
  file.println(" };");

  file.print("const float myOutput_b[] = { ");
  for(int i=0; i<3; i++) {
    file.print(myDense_b[i], 6); file.print("f");
    if(i < 2) file.print(",");
  }
  file.println(" };");

  file.println("\n#endif");
  file.close();
  return true;
}

// SETUP
void setup() {
  Serial.begin(115200);
  pinMode(A0, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  delay(2000);
  
  if(!SD.begin(21)) {
    Serial.println("[ERR] SD init failed. Check connection.");
  } else {
    myLoadImagesFromSd();
  }
  
  if(!myAllocateModelMemory()) while(1) delay(1000);
  
  Serial.println("Press A0 to start...");
}

// LOOP
void loop() {
  if(analogRead(A0) > 2000) {
    digitalWrite(LED_BUILTIN, LOW);
    
    // Check if SD is active before proceeding with training steps that need it
    if(!SD.begin(21)) {
        Serial.println("no sd card");
    } else {
        myInitializeWeights();
        myTrainModel();
        if(mySaveModelHeader()) {
            Serial.println("Model saved successfully.");
        } else {
            Serial.println("Model save FAILED.");
        }
        myPrintDebug();
    }
    
    digitalWrite(LED_BUILTIN, HIGH);
    delay(2000);
  }
  delay(100);
}