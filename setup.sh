pip install -r requirements.txt

curl \
-H "Authorization: Bearer ya29.a0Ad52N39CRtuIJJ4aPM72acn3Bbr4CKPv1RkgJt2EWec6A4jjW86TllEhQU4T4mXdz1dWdJ1zRmoUYkvm1n0hXs6W4ga5HQl9Fse5XyN2-p0KeDMrVJEkh57LGsVnadYMj_csQ8W0KgIWBDjQJF7HJeVPPnvqkxk1FYV9aCgYKAVkSARASFQHGX2MilC9cmzcl-bQM_OTEQg341A0171" \
https://www.googleapis.com/drive/v3/files/1Xol4SUBhzZoPh0Q0eCqtUdhCbFvDmYeW?alt=media \
-o Resnet50_Final.pth


apt-get update && apt-get install ffmpeg libsm6 libxext6  -y