# Automatical-Kidney-Segmentation

## 1. Project Goal
The goal of the project is to automatically segment out all the kidneys of a CT scan set. Given the set number and the number of slices in which both of the kidneys appear and disappear, the project can start from the middle slice and automatically segment out the kidneys untill they disappear. The images for this experiment are all from seattle childrens' hospital. For the concerning of privacy, the original images will not be showed in public.

## 2. Steps

### 2.1 Cavity Boundary Detection ###
Discard pixels outside the abdominal boundary, and use thresholding first, then using canny edge detector to find contours, find the largest contours, and fit ellipse of that contour.
