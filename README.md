# Video Facial Expression Analysis

## 1. Methodology

Video Input → Frame Extraction → Face Detection → Expression Prediction → Frame Analysis → Result Generation

## 2. Description

The Video Facial Expression Analysis System analyzes video frames, detects human faces, predicts facial expressions using a pre-trained Hugging Face model, and generates an annotated video with a summary of predictions.

**Technologies Used:** Python, OpenCV, Hugging Face Transformers, PyTorch.

## 3. Input / Output

**Input:** `input.mp4` (approximately 2 minutes).

**Output:**
- Annotated video: `output_frame_summary.mp4`
- CSV report: `frame_emotion_summary.csv`

### Output Screenshots

**Output 1**

![Output Screenshot 1](images/output%201.png)

**Output 2**

![Output Screenshot 2](images/output%202.png)

## 4. Example Output

**Frame-Level Summary:**
- Happy = 561
- Sad = 74
- Neutral = 14
- Uncertain = 595
- No Face = 1600

**Individual Face Predictions:**
- Happy = 1096
- Neutral = 66
- Sad = 160
- Uncertain = 791

**Total Analyzed Frames:** 2844

**Total Face Predictions:** 2113

**Processing Time:** 10.89 minutes

*Note: Predictions are experimental and do not represent model accuracy.*

## 5. GitHub Repository

[Video Facial Expression Analysis](https://github.com/Ch6785/video-facial-expression-analysis)