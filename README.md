# Facial-Recognition-Program

A python program that makes use of the pre-existing face_recognition library (by Adam Geitgey) to allow the user to train the program to recognize faces and then identify who those faces belong to in any image. 
The program is run using a command line interface or a simple GUI and draws a relevantly named bounding box around each face it finds that it has trained on and then renames the image to contain the names 
of the people contained within.
The facial recognition is achieved by taking a series of measurement (encodings) on faces in the training images and then comparing those measurements with the measurements taken from images with unknown faces.

Running the program should create three folders in the python path directory. The Training folder is where you should place the names and images you wish to train the program on. For each person create a new folder within the Training folder, name it with the persons first name and last name: "Firstname_Lastname" and place training photos of the person within this new folder. The more photos of the person the more accurate the program will be. The Output folder is where the file containing the encodings will be stored after the program finished training. The Validation folder is where you can place photos of the people you have trained the program on and then use the validation command to test that the program is working correctly. Photos selected for facial recognition will be renamed and output into the python path directory. 
