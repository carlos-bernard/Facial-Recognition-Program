"""Python file used to train on new faces by generating encodings on training images"""
import pickle
from pathlib import Path
import face_recognition

"defines a constant for the default encoding path"
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

"""creates directories for training photos, validation photos and encodings if they are not present."""
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

"""assigns the model to be used to locate faces in images. In this case "hog" or histogram of orientated gradients.
loops through folders in training directory, saves the folder label into name variable, 
loads each image and builds dictionary of names + encodings"""
def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    print("Generating encodings on images in training file.....")
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        """detects the location of the faces in an image in the form of a tuple storing coordinates of a square around
         the face"""
        face_locations = face_recognition.face_locations(image, model=model)
        """generates encodings(numeric representation of facial features) for each detected face"""
        face_encodings = face_recognition.face_encodings(image, face_locations)
        """adds generates names and encodings to lists"""
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    """saves encodings to the output folder"""
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

