import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import yagmail
from keras.models import load_model

fc=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt.xml"
lc=os.path.dirname(cv2.__file__)+"/data/haarcascade_lefteye_2splits.xml"
rc=os.path.dirname(cv2.__file__)+"/data/haarcascade_righteye_2splits.xml"
face = cv2.CascadeClassifier(fc)
leye = cv2.CascadeClassifier(lc)
reye = cv2.CascadeClassifier(rc)
lbl = ['Close', 'Open']
model = load_model('models/cnncat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
rpred = [99]
lpred = [99]
color = (0, 255, 0)

def detect_faces(image,ans):
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
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Sleep Detector</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    ans=''
    image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        result_img, ans=detect_faces(image,ans)
        dat = "The Eyes are " + ans
        st.image(result_img)
        st.download_button("Download Analysis", dat, file_name='Processed_Image.txt', key='Download Image Analysis')
    rec = st.text_input('Enter your email')
    if st.button("Mail to me"):
          result_img, ans = detect_faces(image, ans)
          dat = "The Eyes are " + ans
          if(len(rec)!=0):
           yag = yagmail.SMTP('devanshkumaravi@gmail.com', 'oowhmqyyreotkwys')
           contents = [dat]
           yag.send(rec, 'subject', contents)
          else:
           st.warning('Please Enter Your Email', icon="⚠️")
if __name__== '__main__':
    main()

