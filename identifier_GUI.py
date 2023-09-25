import wx
import face_identifier
import train_identifier

"Global variable that holds the path to the image the user wants to identify so it can be passed to recognize_faces()"
global unknown_face

"Code to create the graphical user interface using the module wx"
class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Recognise Faces', size=(375, 215))
        panel = wx.Panel(self)
        my_sizer = wx.BoxSizer(wx.HORIZONTAL)
        vbox = wx.BoxSizer(wx.VERTICAL)

        "Creates a text item to act as a title banner for the GUI"
        text_item = wx.StaticText(panel, label="\nFacial Recognition Program",
                                  size=((180,50)), pos=((100,10)), style=wx.ALIGN_CENTER | wx.SIMPLE_BORDER)

        vbox.Add(text_item, 0, wx.ALL | wx.ALIGN_TOP)
        "set the visual style of the text."
        text_item.SetBackgroundColour(wx.BLACK)
        text_item.SetForegroundColour(wx.WHITE)

        "Creates the button for training the program on a new set of images"
        train_button = wx.Button(panel, label='Train')
        train_button.Bind(wx.EVT_BUTTON, train_btn_press)
        my_sizer.Add(train_button, 0, wx.ALL | wx.CENTER, 5 )

        "Creates the button used to validate that the program has trained properly on its training images and works."
        valid_button = wx.Button(panel, label='Validate')
        valid_button.Bind(wx.EVT_BUTTON, valid_btn_press)
        my_sizer.Add(valid_button, 0, wx.ALL | wx.CENTER, 5)

        "Creates the button used to choose a photo to identify faces in."
        identify_button = wx.Button(panel, label='Choose Photo')
        identify_button.Bind(wx.EVT_BUTTON, handler=self.onOpenFile)
        my_sizer.Add(identify_button, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(my_sizer)

        "Creates the button used to identify the face in the chosen image."
        identify_button2 = wx.Button(panel, label='Identify')
        identify_button2.Bind(wx.EVT_BUTTON, identify_btn_press)
        my_sizer.Add(identify_button2, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(my_sizer)
        self.Show()

    """
    UI element that allows the user to select an image file then returns the path of that image file
    to the unknown_face global variable.
    """
    def onOpenFile(self, event):
        global unknown_face

        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            wildcard="pictures (*.jpeg,*.png,*jpg)|*.jpeg;*.png;*.jpg",
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
            )

        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following file:")
            for path in paths:
                print(path)
                unknown_face = str(path)

        dlg.Destroy()


"Code for button press events that calls functions from face_identifier.py and train.py"
def valid_btn_press(event):
    face_identifier.validate()


def train_btn_press(event):
    train.encode_known_faces()


def identify_btn_press(event):
    face_identifier.recognize_faces(unknown_face)

"Code to protect the user from invoking the script when they didnt intend to"
if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
