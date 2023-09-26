import pickle
from pathlib import Path
import face_recognition
import argparse
import os
import time
import train_identifier
from PIL import Image, ImageDraw
from collections import Counter

"""
A python program that makes use of the pre-existing face_recognition library (by Adam Geitgey) to allow the user to 
train the program to recognize
faces and then identify who those faces belong to in any image. The program is run using a command line interface or a 
simple GUI and draws a relevantly named bounding box around each face it finds that it has trained on and then 
renames the image to contain the names of the people contained within.
The facial recognition is achieved by taking a series of measurement (encodings) on faces in the training images and 
then comparing those measurements with the measurements taken from images with unknown faces
"""

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

"""builds a command line interface to access functionality of the program. Works as an alternative to the GUI built in
the interface.py module"""
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true", help="Trains on a set of images")
parser.add_argument(
    "-v", "--validate", action="store_true", help="validates by taking known faces and trying to identify them")
parser.add_argument(
    "-i", "--identify", action="store_true", help="specify location of image with unknown faces to be identified")
parser.add_argument("-f", action="store", help="path to image to be identified")
args = parser.parse_args()

"""
Is passed location of images with known and named faces, model to be used and location to save encodings and then 
generates them.
Encodings are a collection of measurements taken of faces used to identify the faces owner.
"""
def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH, ) -> None:
    new_file_name = ""
    """opens the encoding file and loads its data with the pickle module"""
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
        """loads the image with the face_recognition library and assigns the output"""
        input_image = face_recognition.load_image_file(image_location)

        """detects faces in the input image and retrieves their encodings using the face_recognition library"""
        input_face_locations = face_recognition.face_locations(
            input_image, model=model)

        input_face_encodings = face_recognition.face_encodings(
            input_image, input_face_locations)

    """creates a pillow image object and imagedraw object to draw a bounding box around detected face"""
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    """uses zip for parallel iteration through input_face_locations and input_face_encodings. calls recognize_face() 
    for encodings of the known and unknown images."""
    for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)

        """assigns unknown if a match isnt found"""
        if not name:
            name = "Unknown"
        else:
            print("image contains " + name + "\n")
            new_file_name = new_file_name + name
        """prints the name and coordinates of the identified faces"""
        _display_face(draw, bounding_box, name)

    """renames the original image file to the name of the people identified in it, keeping the same file extension."""
    orig_name = image_location
    extension = orig_name.split(".")[-1]
    date = time.strftime('%H:%M:%S')
    date = date.replace(":", "")
    os.rename(os.path.join(os.getcwd(), orig_name), new_file_name + "_" + date + "." + extension)

    """removes draw object from scope and shows the image"""
    del draw
    pillow_image.show()


"""uses compare_faces() from the face_recognition library to compare the encodings of the unknown face to the
encodings taken from encode_known_faces()."""
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding)
    """Counter() counts these returns and every true value counts as a vote towards the closest match"""
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    """returns the name of the encodings with the most votes and is therefore the closest match."""
    if votes:
        return votes.most_common(1)[0][0]


"""unpacks the bounding_box tuple into its top,right, etc coordinates and uses them to draw a box around the 
recognized face. Creates a box containing identified name anchored to the bounding box"""
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="black",
        outline="black",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )


"""checks accuracy by running recognize_faces on images with known faces in the validation directory to make sure 
the known encodings are accurate"""
def validate(model: str = "hog"):
    """uses pathlib to open the validation directory and .rglob() to open all the files inside"""
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)


if __name__ == "__main__":
    if args.train:
        train_identifier.encode_known_faces()
    if args.validate:
        validate()
    if args.identify:
        recognize_faces(image_location=args.f)
        print(args.f)
