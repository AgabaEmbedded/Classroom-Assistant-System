import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
import time
import socket
from collections import Counter
import json
from streamlit_option_menu import option_menu
from retinaface import RetinaFace
import mediapipe as mp
from mtcnn import MTCNN
st.set_page_config(page_title="Classroom Assistant",page_icon="📝",layout='wide')
#from page.model_loader import load_model

# Directory to save the dataset
attendance = r"student_data\attendance"
course_list_path = r'student_data\course list.json'
faces = r'student_data\faces'
#try:
#    model = tf.keras.models.load_model(r'Latest FER.keras')
#except Exception as e:
#    print('error')
class_names=['anger', 'calm', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

raspi_ip = '192.168.43.230'  # Replace with your Raspberry Pi's IP address
raspi_port = 8000
def detect_and_extract_faces_mediapipe(img_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(img_rgb)
    
    face_results = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x2, y2 = x + w, y + h
            cropped_face = img_rgb[y:y2, x:x2]
            face_results.append({
                'facial_area': [x, y, x2, y2],
                'cropped_face': cropped_face
                            })
    return face_results

def receive_image(connection):
    # First, get the size of the image (4 bytes)
    image_size = int.from_bytes(connection.recv(4), byteorder='big')
    
    # Receive the image in chunks
    image_data = b''
    while len(image_data) < image_size:
        chunk = connection.recv(4096)
        if not chunk:
            break
        image_data += chunk
    return image_data

def get_image():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as laptop_socket:
            laptop_socket.connect((raspi_ip, raspi_port))
            print("Connected to Raspberry Pi, receiving image...")

            # Receive image data
            image_data = receive_image(laptop_socket)

            print('image recieve digitally')
            # Save the image to a file
            with open('received_image.jpg', 'wb') as f:
                f.write(image_data)
                print('image recieved')
            
            #print("Image received and saved as 'received_image.jpg'.")
    except Exception as e:
        print(f"Error: {e}")

def get_emotion(model):
    img = tf.keras.utils.load_img(
        f'face.jpg', target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)[0]
    return class_names[np.argmax(predictions)]

def process_emotions(emotions_list):
    if emotions_list:
        emotion_count = Counter(emotions_list)
        total_count = len(emotions_list)
        emotion_percentage = {emotion: (count / total_count) * 100 for emotion, count in emotion_count.items()}
        dfe = pd.DataFrame(list(emotion_percentage.items()), columns=['Emotion', 'Percentage'])
        df_sorted = dfe.sort_values(by='Percentage', ascending=True)
        #st.subheader("Emotion Percentage Table")
        #st.dataframe(df_sorted)
        fig, ax = plt.subplots(figsize=(3, 1))
        ax.barh(df_sorted['Emotion'], df_sorted['Percentage'], color='skyblue')
        ax.set_xlabel('Percentage', fontsize=4)
        ax.set_ylabel('Emotion', fontsize=4)
        ax.set_title('Percentage Dominance of Each Emotion', fontsize=4)
        ax.tick_params(axis='both', labelsize=4)

        # Display the chart in Streamlit
        st.pyplot(fig)

def increment_if_int(val):
    if isinstance(val, int) and val > 0:
        return val + 1
    return val

def update_final_time(val):
    if isinstance(val, int) and val > 0:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return val
    return 
       
def get_predominant_element(lst):
    return Counter(lst).most_common(1)[0][0]

def load_dataframe(title):
    file_path = os.path.join(attendance, f"{title}.csv")
    return pd.read_csv(file_path)

def save_dataframe(df, title):
    df.to_csv(os.path.join(attendance, f"{title}.csv"), index=False)

def detect_and_extract_faces(img_path, threshold=0.5, align=True):
    # Load the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_details = RetinaFace.detect_faces(img_path)
    face_results = []
    for key, face_data in face_details.items():
        facial_area = face_data["facial_area"]
        x, y, w, h = facial_area[0], facial_area[1], facial_area[2], facial_area[3]
        cropped_face = img_rgb[y:h, x:w]
        face_results.append({
            'facial_area': facial_area,
            'cropped_face': cropped_face
        })
    return face_results


    
def detect_and_extract_faces_mtcnn(img_path, threshold=0.5):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        detections = detector
    except NameError:
        detector = MTCNN()
        detections = detector.detect_faces(img_rgb)
    face_results = []

    for face_data in detections:
        confidence = face_data['confidence']
        
        if confidence >= threshold:
            x, y, w, h = face_data['box']
            x2, y2 = x + w, y + h
            cropped_face = img_rgb[y:y2, x:x2]
            face_results.append({
                'facial_area': [x, y, x2, y2],
                'cropped_face': cropped_face,
            })

    return face_results

def class_end(class_on_placeholder, class_capture_placeholder, emotion_placeholder, df, column_name, title1):
    with class_on_placeholder:
        st.subheader("class ended")
    with class_capture_placeholder:
        st.subheader("")
    with emotion_placeholder:
        st.subheader("")     
    print(f'dataframe before encoding: {df}')
    df[column_name] = df[column_name].apply(update_final_time)
    print(f'this is the dataframe before saving {df}')
    save_dataframe(df, title1)


def start_class():
    global faces
    global course_list_path
    global attendance
    st.write(
                    """
                    <div style="background-color: #4CAF50; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 46px;">
                        Classroom Assistant - Start Class
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
          
    st.write('')
    with open(course_list_path, 'r') as f:
            my_list = json.load(f)
        
    title1 = st.selectbox('Select Course', my_list)

    st.write("Please enter the class duration in minutes.")
    class_duration = st.number_input("Class Duration", min_value=1, step=1)

    if st.button("Start Class"):
        model = tf.keras.models.load_model('Latest FER.keras')
        attendance_path = os.path.join(attendance, f'{title1}.csv')
        

        face_path = os.path.join(faces, title1)
        
        class_on_placeholder = st.empty()
        emotion_placeholder = st.empty()
        class_capture_placeholder = st.empty()
        stop_class_placeholder = st.empty()
                
        if not os.listdir(face_path):
            st.subheader('No Student Enrolled')
        else:
            if class_duration:
                st.session_state.class_active = True
                start_time = datetime.now()
                end_time = start_time + pd.Timedelta(minutes=class_duration)
                
                df = load_dataframe(title1)
                column_name = start_time.strftime("%Y-%m-%d %H:%M:%S")
                df[column_name] = 0
                count = 0

                while datetime.now() < end_time:
                    emotionss=[]
                    with class_on_placeholder:
                        st.subheader("Class is on...")

                    get_image()
                    
                    image = cv2.imread(r'received_image.jpg')
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    if image is not None:
                        #extract_face()
                        #try:
                        print('1')
                        faces = detect_and_extract_faces_mtcnn(r"received_image.jpg") 
                        print('2')
                        if not faces:
                            print('No face Detected') 
                        else:                   
                            for face in faces:
                                rgb_image = Image.fromarray(face['cropped_face'])
                                rgb_image.save('face.jpg')
                                                                
                                cv2.rectangle(image, (face['facial_area'][0], face['facial_area'][1]), (face['facial_area'][2], face['facial_area'][3]), (0, 255, 0), 2)

                                try:
                                    result=DeepFace.find(img_path='face.jpg', db_path=face_path, model_name='Facenet', enforce_detection=False, threshold=0.6)                                  
                                    if not result or result[0].identity.empty:
                                        print("No matching identity found.")
                                    else:
                                        # Proceed if the DataFrame is not empty
                                        matric_number = result[0].identity.iloc[0].split('\\')[-1].split('.')[0].replace('-', '/')
                                        
                                        df.loc[df["Matriculation Number"] == matric_number, column_name]=1
                                        print(f'daframe after incrementing: {df}')

                                except ValueError as e:
                                    print(f'detected face not found in database {e}')   
                                emotionss.append(get_emotion(model))
                                os.remove('face.jpg')
                        with emotion_placeholder:
                            process_emotions(emotionss)
                        with class_capture_placeholder:
                            st.image(image)
                        # Apply the function to the 'value' column
                        df[column_name] = df[column_name].apply(increment_if_int)
                        df.loc[df[column_name]==7, column_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")                
                    else:
                        st.subheader("Error getting image....")
                    with stop_class_placeholder:
                        col1, col2, col3 = st.columns(3)
                        with col2:
                            st.button('End Class', key = count, use_container_width=True, on_click=class_end, args=(class_on_placeholder, class_capture_placeholder, emotion_placeholder, df, column_name, title1))
                    count+=1
                
                class_end(class_on_placeholder, class_capture_placeholder, emotion_placeholder, df, column_name, title1)

            else:
                st.error("Please enter a valid class duration.")

















# Function to check if the student has attended 75% of classes
def check_attendance(attendance_df):
    attendance_df['Total_Present'] = attendance_df.apply(lambda row: sum([1 for x in row[2:] if x != 0]), axis=1)
    attendance_df['Class_Count'] = len(attendance_df.columns) - 2  # Exclude 'Name' column
    attendance_df['Attendance_Status'] = attendance_df['Total_Present'] / attendance_df['Class_Count'] >= 0.75
    attendance_df['Attendance_Status'] = attendance_df['Attendance_Status'].apply(lambda x: 1 if x else 0)
    attendance_df.drop(['Total_Present', 'Class_Count'], axis = 1)
    return attendance_df

# Function to perform face detection and recognition
def detect_and_label_faces(image, attendance_df, title):
    global faces_path
    # Detect faces
    try:
        facess = detect_and_extract_faces_mtcnn(image)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for face in facess:
            cv2.imwrite('face.jpg', face['cropped_face'])
            cv2.rectangle(image, (face['facial_area'][0], face['facial_area'][1]), (face['facial_area'][2], face['facial_area'][3]), (0, 255, 0), 2)

            # Match face with the database
            try:
                result=DeepFace.find(img_path='face.jpg', db_path=os.path.join(faces, title), model_name='Facenet512', enforce_detection=False)  
                print('Face detected')
                
                if not result or result[0].identity.empty:
                    print("No matching identity found.")
                    label = 'Impersonation'
                else:
                # Proceed if the DataFrame is not empty
                    print(f'the dataframe of face found {result[0]}')
                    matric_number = result[0].identity.iloc[0].split('\\')[-1].split('.')[0].replace('-', '/')
                    student_row = attendance_df[attendance_df['Matriculation Number'] == matric_number]
                    attendance_status = student_row.iloc[0]['Attendance_Status']
                    label = f'{matric_number}: {"Present" if attendance_status == 1 else "Absent"}'
                    

            except ValueError as e:
                print(f'face not found {e}')
                label = 'Impersonation'

            # Draw the bounding box and label
            
            cv2.putText(image, label, (face['facial_area'][0], face['facial_area'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return image
    except ValueError as e:
        return None

def exam_mode():
    # Streamlit UI
    st.write(
                        """
                        <div style="background-color: #4CAF50; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 46px;">
                            Classroom Assistant - Exam Mode
                        </div>
                        """,
                        unsafe_allow_html=True
                        )

    st.write("")
    with open(course_list_path, 'r') as f:
            my_list = json.load(f)
        
    title1 = st.selectbox('Select Course', my_list)

    st.write("Please enter the exam duration in minutes.")
    exam_duration = st.number_input("Exam Duration", min_value=1, step=1)

    if st.button("Commence Exam"):
        df = pd.read_csv(os.path.join(attendance, f'{title1}.csv'))

        # Compute 75% attendance
        df = check_attendance(df)
        print(f'"Updated Attendance Data with 75% threshold: {df}')
        exam_on_placeholder = st.empty()
        class_capture_placeholder = st.empty()
        no_face_placeholder = st.empty()
        stop_exam_placeholder = st.empty()
        if exam_duration:
                start_time = datetime.now()
                end_time = start_time + pd.Timedelta(minutes=exam_duration)
                with exam_on_placeholder:
                    st.subheader("Examination is ongoing...")
                count = 0
                while datetime.now() < end_time:
                    
                    get_image()
                    print('1')

                    processed_img = detect_and_label_faces(r'received_image.jpg', df, title1)
                    print('2')

                    if processed_img is not None:
                        print('3')
                        # Display the image with bounding boxes
                        with class_capture_placeholder:
                            st.image(processed_img, caption="Class Capture with Attendance Status", use_column_width=True)
                        print('4')
                        
                    else:
                        image = cv2.imread(r'received_image.jpg')
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        with no_face_placeholder:
                            st.subheader("No face detected")
                        with class_capture_placeholder:
                            st.image(image, caption="Class Capture with Attendance Status", use_column_width=True)
                        
                        print('5')
                    with stop_exam_placeholder:
                        col1, col2, col3 = st.columns(3)
                        with col2:
                            st.button('End Examination', key = count, use_container_width=True)
                            #end_time=datetime.now()

                    count+=1

        else:
            st.error("Please enter a valid class duration.")













# pages/enroll_student.py

def get_image_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    ret, frame = cap.read()

    if ret:
        image_path = 'received_image.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
    else:
        print("Error: Could not capture image.")

    cap.release()
    cv2.destroyAllWindows()

def enroll_student():
    st.write(
                    """
                    <div style="background-color: #4CAF50; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 40px;">
                        Classroom Assistant - Enroll Student
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
    st.write("")

    st.subheader("Enter the student's details and upload their face image.")
    
    with open(course_list_path, 'r') as f:
            my_list = json.load(f)
        
    title1 = st.selectbox('Select Course', my_list)

    name = st.text_input("Name")
    matric_number = st.text_input("Matriculation Number")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

    if st.button("Enroll"):
        if name and matric_number:
            if not uploaded_file:
                get_image_cam()
                face = detect_and_extract_faces_mediapipe('received_image.jpg')
                st.image(face[0]['cropped_face'])
                rgb_image = Image.fromarray(face[0]['cropped_face'])
                rgb_image.save('face.jpg')

                image = Image.open('face.jpg').convert("RGB").resize((255, 255))
            else:
                # Use the uploaded file
                uploaded_file = Image.open(uploaded_file)
                uploaded_file.save('received_image.jpg')
                face = detect_and_extract_faces_mediapipe('received_image.jpg')
                rgb_image = Image.fromarray(face[0]['cropped_face'])
                rgb_image.save('face.jpg')

                image = Image.open('face.jpg').convert("RGB").resize((255, 255))

            # Save the image
            save_path = os.path.join(faces, title1, (matric_number.replace("/", "-") + '.jpg'))
            image.save(save_path)

            # Display the image
            st.image(image, caption="Uploaded Face")
            
            # Load and update DataFrame
            df = load_dataframe(title1)
            new_entry = pd.DataFrame({
                "Name": [name],
                "Matriculation Number": [matric_number]
            })
            
            df = pd.concat([df, new_entry], ignore_index=True)
            save_dataframe(df, title1)
            
            st.success("Student added successfully!")
        else:
            st.error("Please fill out all fields and upload an image.")












# pages/remove_student.py
def remove_student():
    st.write(
                    """
                    <div style="background-color: #4CAF50; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 40px;">
                        Classroom Assistant - Remove Student
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
    st.write("")
    st.subheader("Enter the details of the student to remove.")

    with open(course_list_path, 'r') as f:
            my_list = json.load(f)
        
    title1 = st.selectbox('Select Course', my_list)

    matric_number = st.text_input("Matriculation Number")

    if st.button("Remove"):
        if matric_number:
            df = load_dataframe(title1)
            if matric_number in df["Matriculation Number"].values:
                df = df[df["Matriculation Number"] != matric_number]
                save_dataframe(df, title1)
                os.remove(os.path.join(faces, title1, (matric_number.replace("/", "-") + '.jpg')))
                st.success("Student removed successfully!")
            else:
                st.error("Student not found.")
        else:
            st.error("Please enter a matriculation number.")











# pages/homepage.py
def create_default():
    course_list = ['first course']
    title = 'first course'
    if os.path.exists(course_list_path):
        pass
    else:
        #os.makedirs(course_list_path)
        with open(course_list_path, 'w') as f:
            json.dump(course_list, f)

    if os.path.exists(attendance):
        pass
    else:
        os.makedirs(attendance)
        file_path = os.path.join(attendance, f"{title}.csv")
        pd.DataFrame(columns=["Name", "Matriculation Number"]).to_csv(file_path, index=False)
    
    if os.path.exists(faces):
        pass
    else:
        file_path = os.path.join(faces, f"{title}")
        os.makedirs(file_path)


def add_course(title):
    with open(course_list_path, 'r') as f:
        my_list = json.load(f)
    my_list.append(title)
    with open(course_list_path, 'w') as f:
        json.dump(my_list, f)

    file_path = os.path.join(attendance, f"{title}.csv")
    pd.DataFrame(columns=["Name", "Matriculation Number"]).to_csv(file_path, index=False)

    file_path = os.path.join(faces, f"{title}")
    os.makedirs(file_path)
    
def homepage():
    st.write(
                    """
                    <div style="background-color: #4CAF50; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 46px;">
                        Classroom Assistant - Homepage
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
    
    st.header("Manage the Attendance and Emotion of your Classroom Efficiently")
    #st.write("What do you want to do today?")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    create_default()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### *CREATE NEW COURSE*')
        new_title = st.text_input('Enter New Course Title')
        if st.button('Create Course'):
            add_course(new_title)

    with col2:
        st.markdown('##### *DOWNLOAD ATTENDANCE*')
        with open(course_list_path, 'r') as f:
            my_list = json.load(f)
        
        title1 = st.selectbox('Select Course', my_list)
        st.download_button("Download Current Attendance", 
                        data = pd.read_csv(os.path.join(attendance, f'{title1}.csv')).to_csv().encode("utf-8"), 
                        file_name=f"## {title1} Attendance.csv",
                    mime = 'text/csv')    














# main.py
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Homepage", "Enroll Student", "Remove Student", "Start Class", "Exam Mode"],
            icons = ['house','arrow-up-square', 'arrow-down-square', 'caret-right-square', 'caret-right-square'],
            default_index=0
        )
    if selected == 'Homepage':
        homepage()
    elif selected == 'Enroll Student':
        enroll_student()
    elif selected == 'Remove Student':
        remove_student()
    elif selected == 'Start Class':
        start_class()
    elif selected == 'Exam Mode':
        exam_mode()

if __name__ == "__main__":
    main()    