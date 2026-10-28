# Video Facial Expression Analysis

## Methodology
- Python
- OpenCV for face detection and video processing
- Hugging Face Transformers for facial expression classification
- Pretrained Vision Transformer (ViT) model
- Frame sampling for efficient video analysis

## Description
This project analyzes a video to detect human faces and predict facial expressions such as happy, sad, angry, neutral, fear, and surprise. It processes video frames, identifies faces, classifies their expressions, and generates a summary of the predictions.

## Input/Output
**Input:** A video file (`input.mp4`), approximately 2 minutes long.

**Output:**
- An annotated video (`output_annotated.mp4`) displaying detected faces, predicted expressions, and confidence scores.
- A terminal summary showing the count of each predicted expression.

The video is processed by analyzing every fifth frame to reduce processing time. The predictions are experimental and may not always be accurate.

## Live Link
Not deployed yet.