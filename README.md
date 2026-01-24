# face_blur_and_emoji
adds blur to all the faces in the video and add emoji based on emotions if requested.

Environment:
conda create -n face python=3.10.13 -y 
conda activate face

🟢 OPTION A — CPU ONLY (simplest, safest)
Install PyTorch (CPU)
pip install torch==2.1.2 torchvision==0.16.2

Install rest
pip install ultralytics==8.1.0
pip install opencv-python
pip install numpy
pip install scipy
pip install matplotlib

🟣 OPTION B — GPU (CUDA 11.8, recommended)
Since you’ve worked with CUDA before, this is ideal.

Install PyTorch + CUDA
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

Install rest
pip install ultralytics==8.1.0
pip install opencv-python
pip install numpy
pip install scipy
pip install matplotlib