# Gesture-recognition-using-CNNLSTM
Using GRIT dataset, built model combining 2D CNN to LSTM to perform real-time gesture recognition from webCam video feed. Built another model employing 3D CNN with LSTM too. 

## Aim
- Build Computer Vision Deep learning model capable of real-time detection of gestures in video.
- Model should be able to run on Low-end devices. (without GPU)
- Model should be quickly trainable (Train with in an 30 min)
- Extracting usable accuracy with limited sample per gesture action.

## Preprosessing
As our motive is motion identification, we first have to detect motion between the sequence of
frames. I use Temporal difference method.

Temporal difference involves difference between two or three successive frames and then
coagulating the difference between successive frames to extract the motion of moving object.
It is very easy and fast to compute and serves better in dynamic environment.

Differential image is computed using the equation:
`
Δ = (𝐼𝑡 − 𝐼𝑡−1) ⋀ (𝐼𝑡+1 − 𝐼𝑡 )
`
where 𝐼𝑡 is frame at time t. Similarly, 𝐼𝑡−1 and 𝐼𝑡+1 are frames at time t-1 (previous time step)
and t+1 (next time step). ⋀ is bitwise AND operator.

## Model Architecture
- Using CNN to extract spatial features from input differential image
- Using LSTM to capture temporal features from sequence of differential images comprising of a complete gesture motion.

### CNN-LSTM model
1. In order to combine CNN with LSTM, I use 22 CNN's run in parallel (Internally run in sequence).
2. Then collate the output of these 22 CNN to form a single input for LSTM
