
# Low-Level Image Stitching from Scratch (C++)

This project implements a multi-image panorama stitching algorithm entirely from scratch in C++, without using high-level OpenCV stitching modules. It focuses on raw matrix and pixel operations.

## Features
* **Harris Corner Detection:** Multi-scale feature extraction using Sobel gradients and Non-Maximum Suppression (NMS).
* **Feature Matching:** Sum of Absolute Differences (SAD) window matching with a 0.8 Ratio Test.
* **RANSAC & Pure Translation Model:** Eliminates outliers and calculates sub-pixel alignment without distortion/stretching artifacts.
* **Inverse Warping & Feathering:** Blends multiple images seamlessly on a large dynamic canvas using weighted pixel accumulation.

## Example Outputs
<img width="1648" height="516" alt="Adim3_Final_Panorama" src="https://github.com/user-attachments/assets/edd1d964-482b-4e7f-948b-8b9169f0f213" />

<img width="1877" height="538" alt="Adim3_Final_Panorama" src="https://github.com/user-attachments/assets/4b406ede-c9e4-4d37-8f23-56cb1476275c" />

<img width="1470" height="532" alt="Adim3_Final_Panorama" src="https://github.com/user-attachments/assets/b4fd1fd2-7f29-44d0-847b-c3f2345edcf0" />

