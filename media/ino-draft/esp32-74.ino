
#include "esp_camera.h"
#include "img_converters.h"
#include "myModel.h" 

// --- 1. CONFIGURATION & BUFFERS ---
#ifdef USE_GRAYSCALE_MODE
  float myInputBuffer[64 * 64 * 1];  
#else
  float myInputBuffer[64 * 64 * 3];  
#endif

float myConv1Output[62 * 62 * 4];
float myPool1Output[31 * 31 * 4];
float myConv2Output[29 * 29 * 8];

inline float clipValue(float val, float minVal = -100.0f, float maxVal = 100.0f) {
    if (isnan(val) || isinf(val)) return 0.0f;
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}

#ifdef USE_INT8_MODE
  #define GET_W(arr, idx, scale) ((float)arr[idx] / scale)
#else
  #define GET_W(arr, idx, scale) (arr[idx])
  float myConv1_w_scale=1, myConv1_b_scale=1;
  float myConv2_w_scale=1, myConv2_b_scale=1;
  float myOutput_w_scale=1, myOutput_b_scale=1;
#endif

// Camera Pins
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM   48
#define Y8_GPIO_NUM   11
#define Y7_GPIO_NUM   12
#define Y6_GPIO_NUM   14
#define Y5_GPIO_NUM   16
#define Y4_GPIO_NUM   18
#define Y3_GPIO_NUM   17
#define Y2_GPIO_NUM   15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// --- 2. LAYERS ---
void myConv1() {
    for (int f = 0; f < 4; f++) {
        int outBase = f * 3844;
        for (int y = 0; y < 62; y++) {
            for (int x = 0; x < 62; x++) {
                float sum = 0;
                #ifdef USE_GRAYSCALE_MODE
                  for (int ky = 0; ky < 3; ky++) {
                      for (int kx = 0; kx < 3; kx++) {
                          int pIdx = (y+ky)*64 + (x+kx);
                          int wIdx = (f*9) + (ky*3) + kx;
                          sum += myInputBuffer[pIdx] * GET_W(myConv1_w, wIdx, myConv1_w_scale);
                      }
                  }
                #else
                  for (int ky = 0; ky < 3; ky++) {
                      for (int kx = 0; kx < 3; kx++) {
                          int pIdx = ((y+ky)*64 + (x+kx))*3;
                          int wIdx = (f*27) + (ky*9) + (kx*3);
                          sum += myInputBuffer[pIdx]   * GET_W(myConv1_w, wIdx,   myConv1_w_scale);
                          sum += myInputBuffer[pIdx+1] * GET_W(myConv1_w, wIdx+1, myConv1_w_scale);
                          sum += myInputBuffer[pIdx+2] * GET_W(myConv1_w, wIdx+2, myConv1_w_scale);
                      }
                  }
                #endif
                sum += GET_W(myConv1_b, f, myConv1_b_scale);
                sum = clipValue(sum, -100.0f, 100.0f);
                // Changed from ReLU to LeakyReLU to prevent dying neurons
                // Debug showed 50% zeros in CONV1 for grayscale, 22% for RGB
                // LeakyReLU allows small negative values: f(x) = x if x>0, else 0.1*x
                myConv1Output[outBase + (y*62 + x)] = (sum > 0) ? sum : (0.1f * sum);
            }
        }
    }
}

void myMaxPool1() {
    for (int f = 0; f < 4; f++) {
        int inBase = f * 3844;
        int outBase = f * 961;
        for (int y = 0; y < 31; y++) {
            for (int x = 0; x < 31; x++) {
                int inY = y * 2; int inX = x * 2;
                float maxVal = myConv1Output[inBase + (inY*62 + inX)];
                maxVal = max(maxVal, myConv1Output[inBase + (inY*62 + inX+1)]);
                maxVal = max(maxVal, myConv1Output[inBase + ((inY+1)*62 + inX)]);
                maxVal = max(maxVal, myConv1Output[inBase + ((inY+1)*62 + inX+1)]);
                myPool1Output[outBase + (y*31 + x)] = maxVal;
            }
        }
    }
}

void myConv2() {
    for (int f = 0; f < 8; f++) {
        int outBase = f * 841;
        for (int y = 0; y < 29; y++) {
            for (int x = 0; x < 29; x++) {
                float sum = 0;
                for (int c = 0; c < 4; c++) {
                    int inBase = c * 961;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int pIdx = inBase + ((y+ky)*31 + (x+kx));
                            int wIdx = (f*36) + (c*9) + (ky*3) + kx;
                            sum += myPool1Output[pIdx] * GET_W(myConv2_w, wIdx, myConv2_w_scale);
                        }
                    }
                }
                sum += GET_W(myConv2_b, f, myConv2_b_scale);
                sum = clipValue(sum, -100.0f, 100.0f);
                // Changed from ReLU to LeakyReLU to prevent dying neurons
                // Debug showed 84% zeros in CONV2 for grayscale models
                // LeakyReLU allows small negative values: f(x) = x if x>0, else 0.1*x
                myConv2Output[outBase + (y*29 + x)] = (sum > 0) ? sum : (0.1f * sum);
            }
        }
    }
}

int myGetWinner() {
    float myLogits[3] = {0, 0, 0};
    int totalFeatures = 29 * 29 * 8;
    for (int i = 0; i < 3; i++) {
        double sum = 0.0; double compensation = 0.0;
        for (int j = 0; j < totalFeatures; j++) {
            double term = (double)myConv2Output[j] * GET_W(myOutput_w, i*totalFeatures + j, myOutput_w_scale);
            double y = term - compensation; double t = sum + y;
            compensation = (t - sum) - y; sum = t;
        }
        myLogits[i] = clipValue((float)sum + GET_W(myOutput_b, i, myOutput_b_scale), -50.0f, 50.0f);
    }
    
    float maxLogit = max(max(myLogits[0], myLogits[1]), myLogits[2]);
    float expSum = exp(myLogits[0]-maxLogit) + exp(myLogits[1]-maxLogit) + exp(myLogits[2]-maxLogit);
    Serial.print("Probs: [");
    for (int i = 0; i < 3; i++) {
        float p = exp(myLogits[i]-maxLogit) / expSum * 100.0f;
        Serial.print(p, 0); Serial.print("%");
        if (i < 2) Serial.print(", ");
    }
    Serial.print("] ");
    int win = (myLogits[1] > myLogits[0]) ? 1 : 0;
    if (myLogits[2] > myLogits[win]) win = 2;
    return win;
}

void myProcessCamera(camera_fb_t *fb) {
    uint8_t *rgb = NULL;
    
    // Handle different pixel formats
    if (fb->format == PIXFORMAT_RGB888) {
        // Already in RGB888 format, no conversion needed
        rgb = fb->buf;
    } else {
        // Need to convert (JPEG or other format)
        rgb = (uint8_t *)ps_malloc(fb->width * fb->height * 3);
        if (!rgb) return;
        if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb)) { 
            free(rgb); 
            return; 
        }
    }
    
    float scaleY = (float)fb->height / 64.0f;
    float scaleX = (float)fb->width  / 64.0f;
    for (int y = 0; y < 64; y++) {
        float srcY = (y + 0.5f) * scaleY - 0.5f; int y0 = (int)srcY; int y1 = min(y0 + 1, (int)fb->height - 1); float dy = srcY - y0;
        for (int x = 0; x < 64; x++) {
            float srcX = (x + 0.5f) * scaleX - 0.5f; int x0 = (int)srcX; int x1 = min(x0 + 1, (int)fb->width - 1); float dx = srcX - x0;
            int idx00 = (y0 * fb->width + x0) * 3; int idx01 = (y0 * fb->width + x1) * 3;
            int idx10 = (y1 * fb->width + x0) * 3; int idx11 = (y1 * fb->width + x1) * 3;
            float r = (1.0f - dy) * ((1.0f - dx) * rgb[idx00] + dx * rgb[idx01]) + dy * ((1.0f - dx) * rgb[idx10] + dx * rgb[idx11]);
            float g = (1.0f - dy) * ((1.0f - dx) * rgb[idx00 + 1] + dx * rgb[idx01 + 1]) + dy * ((1.0f - dx) * rgb[idx10 + 1] + dx * rgb[idx11 + 1]);
            float b = (1.0f - dy) * ((1.0f - dx) * rgb[idx00 + 2] + dx * rgb[idx01 + 2]) + dy * ((1.0f - dx) * rgb[idx10 + 2] + dx * rgb[idx11 + 2]);
            #ifdef USE_GRAYSCALE_MODE
              float gray = (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
              myInputBuffer[y * 64 + x] = gray / 255.0f; 
            #else
              int baseIdx = (y * 64 + x) * 3;
              myInputBuffer[baseIdx] = r / 255.0f; myInputBuffer[baseIdx + 1] = g / 255.0f; myInputBuffer[baseIdx + 2] = b / 255.0f;
            #endif
        }
    }
    
    // Free RGB buffer only if we allocated it (not if using RGB888 directly)
    if (fb->format != PIXFORMAT_RGB888) {
        free(rgb);
    }
}

void setup() {
    Serial.begin(115200);
    
    // Print model configuration at startup
    Serial.println("\n========== MODEL CONFIGURATION ==========");
    #ifdef USE_GRAYSCALE_MODE
      Serial.println("Color Mode: GRAYSCALE (1-channel)");
      Serial.print("Input Buffer Size: 64x64x1 = ");
      Serial.println(64*64*1);
    #else
      Serial.println("Color Mode: RGB (3-channel)");
      Serial.print("Input Buffer Size: 64x64x3 = ");
      Serial.println(64*64*3);
    #endif
    
    #ifdef USE_INT8_MODE
      Serial.println("Quantization: INT8");
      Serial.print("Conv1_w scale: "); Serial.println(myConv1_w_scale, 6);
      Serial.print("Conv2_w scale: "); Serial.println(myConv2_w_scale, 6);
      Serial.print("Output_w scale: "); Serial.println(myOutput_w_scale, 6);
    #else
      Serial.println("Quantization: FLOAT32");
    #endif
    
    Serial.println("\nClass Labels:");
    Serial.print("  0: "); Serial.println(myClassLabels[0]);
    Serial.print("  1: "); Serial.println(myClassLabels[1]);
    Serial.print("  2: "); Serial.println(myClassLabels[2]);
    
    Serial.println("\nModel Architecture:");
    Serial.println("  Conv1: 3x3 filters, 4 outputs -> 62x62x4");
    Serial.println("  MaxPool: 2x2 -> 31x31x4");
    Serial.println("  Conv2: 3x3 filters, 8 outputs -> 29x29x8");
    Serial.println("  Dense: 6728 -> 3 classes");
    
    Serial.print("\nTotal Parameters: ");
    #ifdef USE_GRAYSCALE_MODE
      int conv1Params = (3*3*1*4) + 4;
    #else
      int conv1Params = (3*3*3*4) + 4;
    #endif
    int conv2Params = (3*3*4*8) + 8;
    int denseParams = (29*29*8*3) + 3;
    Serial.println(conv1Params + conv2Params + denseParams);
    
    Serial.println("=========================================\n");
    Serial.println("DEBUG: Hold A0 high (>2000) for detailed frame analysis");
    Serial.println("Starting inference loop...\n");
    
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM; config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM; config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = -1; config.pin_reset = -1; config.xclk_freq_hz = 10000000;
    config.frame_size = FRAMESIZE_QVGA; 
    
    // Try RGB888 first (uncompressed, best quality)
    // If memory issues occur, fall back to JPEG
    config.pixel_format = PIXFORMAT_RGB888;  // Changed from JPEG to RGB888 for better quality
    
    config.grab_mode = CAMERA_GRAB_LATEST; config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 1; config.jpeg_quality = 12;  // jpeg_quality ignored for RGB888
    
    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("RGB888 failed, trying JPEG...");
        config.pixel_format = PIXFORMAT_JPEG;  // Fallback to JPEG
        esp_camera_init(&config);
    } else {
        Serial.println("Camera initialized with RGB888 (uncompressed)");
    }
}

void loop() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return;
    myProcessCamera(fb);
    esp_camera_fb_return(fb);

    if (analogRead(A0) > 2000) {
        Serial.println("\n========== DETAILED DEBUG ==========");
        
        // 1. MODEL CONFIG REMINDER
        Serial.print("Mode: ");
        #ifdef USE_GRAYSCALE_MODE
          Serial.print("GRAYSCALE ");
        #else
          Serial.print("RGB ");
        #endif
        #ifdef USE_INT8_MODE
          Serial.println("INT8");
        #else
          Serial.println("FLOAT32");
        #endif
        
        // Show camera format
        Serial.print("Camera Format: ");
        if (fb->format == PIXFORMAT_RGB888) Serial.println("RGB888 (uncompressed)");
        else if (fb->format == PIXFORMAT_JPEG) Serial.println("JPEG (compressed)");
        else if (fb->format == PIXFORMAT_RGB565) Serial.println("RGB565");
        else if (fb->format == PIXFORMAT_GRAYSCALE) Serial.println("GRAYSCALE");
        else Serial.println("UNKNOWN");
        
        // 2. INPUT BUFFER STATS
        float minVal = 10.0, maxVal = -10.0, avgVal = 0.0;
        int totalPixels = 64 * 64 * (
          #ifdef USE_GRAYSCALE_MODE
            1
          #else
            3
          #endif
        );
        for (int i = 0; i < totalPixels; i++) {
            float val = myInputBuffer[i];
            if (val < minVal) minVal = val; 
            if (val > maxVal) maxVal = val; 
            avgVal += val;
        }
        avgVal /= totalPixels;
        Serial.print("INPUT - Min: "); Serial.print(minVal, 4); 
        Serial.print(" Max: "); Serial.print(maxVal, 4); 
        Serial.print(" Avg: "); Serial.print(avgVal, 4);
        Serial.print(" Range: "); Serial.print(maxVal - minVal, 4);
        // NOTE: Min should be close to 0.0 only if scene has pure black pixels
        // Typical indoor scenes have ambient light, so min ~0.20-0.25 is normal
        if (minVal > 0.3f) Serial.print(" [HIGH - check lighting/exposure]");
        Serial.println();

        // 3. LAYER STATISTICS
        auto printLayerStats = [](const char* name, float* buf, int len) {
            float mi = 1e6, ma = -1e6, av = 0; 
            int z = 0, neg = 0, clipped = 0;
            for(int i=0; i<len; i++){
                float v = buf[i]; 
                if(v < mi) mi = v; 
                if(v > ma) ma = v; 
                av += v; 
                if(v == 0.0f) z++;
                if(v < 0.0f) neg++;
                if(v == -100.0f || v == 100.0f) clipped++;
            }
            av /= len;
            Serial.print(name); 
            Serial.print(" - Min: "); Serial.print(mi, 2); 
            Serial.print(" Max: "); Serial.print(ma, 2); 
            Serial.print(" Avg: "); Serial.print(av, 2);
            Serial.print(" Neg: "); Serial.print(neg); 
            Serial.print("/"); Serial.print(len); 
            Serial.print(" ("); Serial.print((neg*100)/len); Serial.print("%)");
            Serial.print(" Zeros: "); Serial.print(z); 
            Serial.print("/"); Serial.print(len); 
            Serial.print(" ("); Serial.print((z*100)/len); Serial.print("%)");
            if(clipped > 0) {
                Serial.print(" CLIPPED: "); Serial.print(clipped);
            }
            Serial.println();
        };
        
        printLayerStats("CONV1", myConv1Output, 62*62*4);
        printLayerStats("POOL1", myPool1Output, 31*31*4);
        printLayerStats("CONV2", myConv2Output, 29*29*8);

        // 4. WEIGHT SAMPLES
        Serial.println("\nWeight Samples:");
        Serial.print("  Conv1_w[0-5]: ");
        for (int i = 0; i < 6; i++) { 
            Serial.print(GET_W(myConv1_w, i, myConv1_w_scale), 4); 
            Serial.print(" "); 
        }
        Serial.println();
        
        Serial.print("  Conv1_b[0-3]: ");
        for (int i = 0; i < 4; i++) { 
            Serial.print(GET_W(myConv1_b, i, myConv1_b_scale), 4); 
            Serial.print(" "); 
        }
        Serial.println();
        
        Serial.print("  Output_w[0-9]: ");
        for (int i = 0; i < 10; i++) { 
            Serial.print(GET_W(myOutput_w, i, myOutput_w_scale), 4); 
            Serial.print(" "); 
        }
        Serial.println();
        
        Serial.print("  Output_b[0-2]: ");
        for (int i = 0; i < 3; i++) { 
            Serial.print(GET_W(myOutput_b, i, myOutput_b_scale), 4); 
            Serial.print(" "); 
        }
        Serial.println();
        
        // 4b. DEEP WEIGHT ANALYSIS
        Serial.println("\n--- Deep Weight Check ---");
        
        // Count Conv1 weights
        #ifdef USE_GRAYSCALE_MODE
          int conv1WeightCount = 3*3*1*4;  // 36 weights
        #else
          int conv1WeightCount = 3*3*3*4;  // 108 weights
        #endif
        Serial.print("Conv1 expected weights: "); Serial.println(conv1WeightCount);
        
        // Sample Conv1 weights at different positions
        Serial.print("Conv1_w[middle 54]: "); Serial.println(GET_W(myConv1_w, 54, myConv1_w_scale), 4);
        Serial.print("Conv1_w[last]: "); Serial.println(GET_W(myConv1_w, conv1WeightCount-1, myConv1_w_scale), 4);
        
        // Conv2 weights
        int conv2WeightCount = 3*3*4*8;  // 288 weights
        Serial.print("Conv2 expected weights: "); Serial.println(conv2WeightCount);
        Serial.print("Conv2_w[0-2]: ");
        for (int i = 0; i < 3; i++) {
            Serial.print(GET_W(myConv2_w, i, myConv2_w_scale), 4); Serial.print(" ");
        }
        Serial.println();
        Serial.print("Conv2_w[last]: "); Serial.println(GET_W(myConv2_w, conv2WeightCount-1, myConv2_w_scale), 4);
        
        // Output weights
        int outputWeightCount = 29*29*8*3;  // 20232 weights
        Serial.print("Output expected weights: "); Serial.println(outputWeightCount);
        Serial.print("Output_w[1000]: "); Serial.println(GET_W(myOutput_w, 1000, myOutput_w_scale), 4);
        Serial.print("Output_w[10000]: "); Serial.println(GET_W(myOutput_w, 10000, myOutput_w_scale), 4);
        Serial.print("Output_w[last]: "); Serial.println(GET_W(myOutput_w, outputWeightCount-1, myOutput_w_scale), 4);

        // 4c. CONVOLUTION SANITY CHECK
        Serial.println("\n--- Conv1 Single Pixel Test ---");
        // Test one pixel computation manually
        float testSum = 0;
        int testY = 10, testX = 10, testFilter = 0;
        #ifdef USE_GRAYSCALE_MODE
          for (int ky = 0; ky < 3; ky++) {
              for (int kx = 0; kx < 3; kx++) {
                  int pIdx = (testY+ky)*64 + (testX+kx);
                  int wIdx = (testFilter*9) + (ky*3) + kx;
                  float pixVal = myInputBuffer[pIdx];
                  float weight = GET_W(myConv1_w, wIdx, myConv1_w_scale);
                  testSum += pixVal * weight;
                  if (ky == 1 && kx == 1) {  // Center pixel
                      Serial.print("  Center pixel: input="); Serial.print(pixVal, 4);
                      Serial.print(" weight="); Serial.println(weight, 4);
                  }
              }
          }
        #else
          for (int ky = 0; ky < 3; ky++) {
              for (int kx = 0; kx < 3; kx++) {
                  int pIdx = ((testY+ky)*64 + (testX+kx))*3;
                  int wIdx = (testFilter*27) + (ky*9) + (kx*3);
                  float r = myInputBuffer[pIdx];
                  float g = myInputBuffer[pIdx+1];
                  float b = myInputBuffer[pIdx+2];
                  float wr = GET_W(myConv1_w, wIdx, myConv1_w_scale);
                  float wg = GET_W(myConv1_w, wIdx+1, myConv1_w_scale);
                  float wb = GET_W(myConv1_w, wIdx+2, myConv1_w_scale);
                  testSum += r*wr + g*wg + b*wb;
                  if (ky == 1 && kx == 1) {  // Center pixel
                      Serial.print("  Center pixel RGB: ["); 
                      Serial.print(r, 3); Serial.print(","); 
                      Serial.print(g, 3); Serial.print(","); 
                      Serial.print(b, 3); Serial.println("]");
                      Serial.print("  Weights RGB: [");
                      Serial.print(wr, 4); Serial.print(",");
                      Serial.print(wg, 4); Serial.print(",");
                      Serial.print(wb, 4); Serial.println("]");
                  }
              }
          }
        #endif
        testSum += GET_W(myConv1_b, testFilter, myConv1_b_scale);
        float testOutput = (testSum > 0) ? testSum : (0.1f * testSum);
        Serial.print("Manual Conv1 output at ["); Serial.print(testY); Serial.print(","); Serial.print(testX);
        Serial.print("] filter 0: "); Serial.print(testOutput, 4);
        Serial.print(" (stored value: "); Serial.print(myConv1Output[testFilter*3844 + testY*62 + testX], 4);
        Serial.println(")");

        // 5. LOGITS BEFORE SOFTMAX
        Serial.println("\nLogits (before softmax):");
        float myLogits[3] = {0, 0, 0};
        int totalFeatures = 29 * 29 * 8;
        for (int i = 0; i < 3; i++) {
            double sum = 0.0; 
            double compensation = 0.0;
            for (int j = 0; j < totalFeatures; j++) {
                double term = (double)myConv2Output[j] * GET_W(myOutput_w, i*totalFeatures + j, myOutput_w_scale);
                double y = term - compensation; 
                double t = sum + y;
                compensation = (t - sum) - y; 
                sum = t;
            }
            myLogits[i] = clipValue((float)sum + GET_W(myOutput_b, i, myOutput_b_scale), -50.0f, 50.0f);
            Serial.print("  Class "); Serial.print(i); 
            Serial.print(" ("); Serial.print(myClassLabels[i]); 
            Serial.print("): "); Serial.print(myLogits[i], 4);
            // NOTE: If all logits are very negative (< -10), model may be undertrained
            // Try: More samples, lower learning rate, or more epochs
            if (myLogits[i] < -10.0f) Serial.print(" [VERY LOW]");
            Serial.println();
        }

        // 6. ASCII PREVIEW
        Serial.println("\n--- ASCII PREVIEW (32x16) ---");
        for (int y = 0; y < 64; y += 4) {
            for (int x = 0; x < 64; x += 2) {
                float val;
                #ifdef USE_GRAYSCALE_MODE
                  val = myInputBuffer[y * 64 + x];
                #else
                  int idx = (y * 64 + x) * 3;
                  val = (myInputBuffer[idx] + myInputBuffer[idx+1] + myInputBuffer[idx+2]) / 3.0f;
                #endif
                if (val > 0.75) Serial.print("#"); 
                else if (val > 0.5) Serial.print("+"); 
                else if (val > 0.25) Serial.print("."); 
                else Serial.print(" ");
            }
            Serial.println();
        }
        
        // 7. MEMORY INFO
        Serial.println("\n--- Memory Usage ---");
        Serial.print("Free Heap: "); Serial.print(ESP.getFreeHeap()); Serial.println(" bytes");
        Serial.print("Free PSRAM: "); Serial.print(ESP.getFreePsram()); Serial.println(" bytes");
        
        Serial.println("========== END DEBUG ==========\n");
        delay(2000);
    }

    myConv1(); myMaxPool1(); myConv2();
    int result = myGetWinner();
    Serial.print("Class: "); Serial.print(result); 
    Serial.print(" ("); Serial.print(myClassLabels[result]); Serial.println(")");
    delay(50);
}