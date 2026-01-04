/*
 * esp01.ino
 * Purpose: Verify that the ESP32-S3 can see and read the weights 
 * exported from torch15.py.
 */

#include "myWeights.h"

// We can calculate the number of elements in the array using sizeof
// This is a great C++ trick for students to learn.
int myConv1_Size = sizeof(myConv1_weight) / sizeof(myConv1_weight[0]);
int myHidden_Size = sizeof(myHidden_weight) / sizeof(myHidden_weight[0]);

void setup() {
  // Always use 115200 for ESP32-S3 to handle high-speed data
  Serial.begin(115200);
  while (!Serial) { delay(10); } // Wait for serial to connect
  
  delay(2000); 
  Serial.println("===============================");
  Serial.println("  ESP32-S3 BRAIN INTEGRITY TEST ");
  Serial.println("===============================");

  // 1. Check Conv1 Weights (The "Internal Eyes")
  Serial.print("Conv1 Weights Found: ");
  Serial.println(myConv1_Size);
  
  // 2. Check Hidden Weights (The "Deep Thoughts")
  Serial.print("Hidden Weights Found: ");
  Serial.println(myHidden_Size);

  Serial.println("\n--- Sampling Weights ---");
  
  // Print the first 3 weights of the brain
  for(int i = 0; i < 3; i++) {
    Serial.print("Weight [");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(myConv1_weight[i], 8); 
  }

  Serial.println("\nIf these numbers match torch15.py, the Bridge is SOLID.");
  Serial.println("Ready to connect the camera in esp02.ino!");
}

void loop() {
  // Stay quiet after the test
}