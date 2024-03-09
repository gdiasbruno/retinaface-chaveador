pip install -r requirements.txt

curl \
-H "Authorization: Bearer" \
https://www.googleapis.com/drive/v3/files/1Xol4SUBhzZoPh0Q0eCqtUdhCbFvDmYeW?alt=media \
-o Resnet50_Final.pth


apt-get update && apt-get install ffmpeg libsm6 libxext6  -y