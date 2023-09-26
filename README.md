# Facial-Recognition-Program

A python program that makes use of the pre-existing face_recognition library (by Adam Geitgey) to allow the user to train the program to recognize faces and then identify who those faces belong to in any image. 
The program is run using a command line interface or a simple GUI and draws a relevantly named bounding box around each face it finds that it has trained on and then renames the image to contain the names 
of the people contained within.
The facial recognition is achieved by taking a series of measurement (encodings) on faces in the training images and then comparing those measurements with the measurements taken from images with unknown faces.

Running the program should create three folders in the python path directory. The Training folder is where you should place the names and images you wish the train the program on.
