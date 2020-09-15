# FaceRec
Facial recognition in a vid stream

Taken from https://artelliq.com/blog/how-to-create-a-custom-webcam-facial-recognition-system-in-python/

Building on Ubuntu 18.04

sudo apt-get install python3-venv cmake

python3.6 -m venv env
source env/bin/activate
(or use requirements)
python3.6 -m pip install --upgrade pip
python3.6 -m pip install numpy
python3.6 -m pip install pandas
python3.6 -m pip install matplotlib
python3.6 -m pip install sklearn
python3.6 -m pip install opencv-python

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake ..
cd ..
cmake --build build
# make sure to be in the dlib dir
python3.6 setup.py install
cd ..
python3.6 -m pip install face_recognition






