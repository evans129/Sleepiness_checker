import streamlit as st
import cv2
import numpy as np
import os
import threading
from PIL import Image
import yagmail
from ai_edge_litert.interpreter import Interpreter


class TFLiteModel:
    """Runs a TensorFlow Lite model through the same predict() call the Keras model used."""

    def __init__(self, path):
        self._interpreter = Interpreter(model_path=path)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._lock = threading.Lock()

    def predict(self, batch):
        samples = np.asarray(batch, dtype=np.float32).reshape((-1, *self._input["shape"][1:]))
        outputs = []
        # Streamlit runs each session on its own thread, and an interpreter isn't thread-safe.
        with self._lock:
            for sample in samples:
                self._interpreter.set_tensor(self._input["index"], sample[np.newaxis])
                self._interpreter.invoke()
                outputs.append(self._interpreter.get_tensor(self._output["index"])[0].copy())
        return np.array(outputs)


@st.cache_resource
def load_model(path):
    return TFLiteModel(path)


def email_credentials():
    """Sender address and Gmail app password from Streamlit secrets, or None when email isn't set up."""
    try:
        sender = st.secrets["email"]
        return sender["user"], sender["app_password"]
    except (KeyError, FileNotFoundError):
        return None


fc=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt.xml"
lc=os.path.dirname(cv2.__file__)+"/data/haarcascade_lefteye_2splits.xml"
rc=os.path.dirname(cv2.__file__)+"/data/haarcascade_righteye_2splits.xml"
face = cv2.CascadeClassifier(fc)
leye = cv2.CascadeClassifier(lc)
reye = cv2.CascadeClassifier(rc)
lbl = ['Close', 'Open']
model = load_model('models/cnnCat2.tflite')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
rpred = [99]
lpred = [99]
color = (0, 255, 0)

def detect_faces(image,ans):
 rpred = [99]
 lpred = [99]
 x, y = 10, 40  # where the label goes when no face or eye is detected
 img=np.array(image.convert('RGB'))
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
 left_eye = leye.detectMultiScale(gray)
 right_eye = reye.detectMultiScale(gray)
 for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
 for (x, y, w, h) in right_eye:
    r_eye = img[y:y + h, x:x + w]
    r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
    r_eye = cv2.resize(r_eye, (24, 24))
    r_eye = r_eye / 255
    r_eye = r_eye.reshape(24, 24, -1)
    r_eye = np.expand_dims(r_eye, axis=0)
    predict_r = model.predict(r_eye)
    rpred = np.argmax(predict_r, axis=1)
    # rpred = model.predict_classes(r_eye)
    if (rpred[0] == 1):
        lbl = 'Open'
    if (rpred[0] == 0):
        lbl = 'Closed'
    break

 for (x, y, w, h) in left_eye:
    l_eye = img[y:y + h, x:x + w]
    l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
    l_eye = cv2.resize(l_eye, (24, 24))
    l_eye = l_eye / 255
    l_eye = l_eye.reshape(24, 24, -1)
    l_eye = np.expand_dims(l_eye, axis=0)
    predict_l = model.predict(l_eye)
    lpred = np.argmax(predict_l, axis=1)
    # lpred = model.predict_classes(l_eye)
    if (lpred[0] == 1):
        lbl = 'Open'
    if (lpred[0] == 0):
        lbl = 'Closed'
    break

 if (rpred[0] == 0 and lpred[0] == 0):
    ans+='Closed'
    cv2.putText(img, "Closed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

 else:
    ans+='Open'
    cv2.putText(img, "Open", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
 return img,ans
def main():
    st.title("Sleepiness Checker")
    html_temp="""
    <body style="background-color:red;">
    <div style="background-color:#ff6f61;padding:10px">
    <h2 style="color:white;text-align:center;">Sleep Detector</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    ans=''
    image = None
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.text("original Image")
        st.image(image)
    else:
     image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
     if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        if image is None:
            st.warning('Take a picture or upload an image first', icon="⚠️")
        else:
            result_img, ans=detect_faces(image,ans)
            dat = "The Eyes are " + ans
            st.image(result_img)
            st.download_button("Download Analysis", dat, file_name='Processed_Image.txt', key='Download Image Analysis')
    credentials = email_credentials()
    if credentials is None:
        st.caption("Emailing results is turned off in this demo. Use Download Analysis instead.")
        return
    rec = st.text_input('Enter your email')
    if st.button("Mail to me"):
          if image is None:
           st.warning('Take a picture or upload an image first', icon="⚠️")
          elif(len(rec)!=0):
           result_img, ans = detect_faces(image, ans)
           dat = "The Eyes are " + ans
           yag = yagmail.SMTP(*credentials)
           contents = [dat]
           yag.send(rec, 'subject', contents)
          else:
           st.warning('Please Enter Your Email', icon="⚠️")
if __name__== '__main__':
    main()

