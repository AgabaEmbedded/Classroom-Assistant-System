# Classroom-Assistant-System 
OPERATIONAL MANUAL
DEVELOPMENT OF AN INTELLIGENT CLASSROOM ASSISTANT SYSTEM 
 By
Sunday Agabaidu Abraham 2018/1/70636CP
DEPARTMENT OF COMPUTER ENGINEERING
SCHOOL OF ELECTRICAL ENGINEERING AND TECHNOLOGY
FEDERAL UNIVERSITY OF TECHNOLOGY MINNA, NIGER STATE

 
INTRODUCTION: The world of technology has benefited greatly from the artificial intelligence field's explosive expansion. Machine learning and deep learning algorithms have seen tremendous success in several applications, including pattern recognition, recommendation systems, classification systems, and others, whereas classical algorithms were unable to satisfy the needs of humans in real time. Human intellect, conduct, and emotion are greatly influenced by emotion. (Singh et al., 2021). Since human emotions may be used to learn a lot, using emotions in human-to-human communication is vital. Of all exchanges, nonverbal communication makes up around 66%. 55% of emotions are visual, 38% are vocal, and 7% are verbal, according to research. Humans utilize facial emotion (FE), one of the non-verbal modalities of communication, to infer cues that would otherwise be missed during an interaction (Prince Awuah Baffour et al., 2022).  Applications for facial emotions can be found in many different fields. Facial emotion recognition is the process of recognizing expressions that represent common emotions like fear, happiness, disgust, etc. It is crucial to human-computer interactions and has applications in digital marketing, gaming, education, customer service, and healthcare..(Khaireddin and Chen, 2021), These days, facial expression detection systems are quite important since they can capture human behavior, emotions, intentions, and so on. The facial expression recognition system that uses deep learning has proven to be superior to the conventional approaches, which are slower and less accurate. (Singh et al., 2021) Machine learning is a burgeoning technology in the field of computer science, projected to have a 90% influence within the next four years. Deep learning, a kind of machine learning, employs artificial neural networks that are algorithms inspired by the structure and functioning of the human brain. A CNN  is a type of deep neural network that employs convolution as its primary mathematical activity(Singh et al., 2021).
Recording and managing attendance is a crucial part of institution of learning, it will normally contribute to the final grading of the student.(Darapaneni et al., 2020). In academic institutions and organizations, attendance is very important for a variety of reasons, including keeping records, assessing students, and promoting consistent attendance. The majority of educational institutions in developing nations have minimum attendance requirements, but because of the difficulties associated with the current system of recording attendance, these requirements are not always met. (S. Saul et al., 2023) Even today many institutions still makes use of the conventional method of collecting and managing attendance which involve the use of paper containing list of students and student will sign against their name on the list as a sign that they are present in the class. this is collated at the end of the class. it is usually coordinated by the lecturer or educator in charge(Elias et al., 2019). However there are issues with this general method. The students' fabrication of a close friend's signature is one of the issues. In addition to being laborious, the attendance method is problematic. An automatic attendance system must be implemented in order to address these issues(Elias et al., 2019) In order to simplify maintenance procedures, several institutions have transitioned to utilizing biometric technologies, such as fingerprint, IRIS, and radio frequency identification (RFID). Nevertheless, these methods still have various drawbacks, such as the lengthy duration they take, the tedious maintenance of class attendance records, the potential for fraudulent attendance, and the ongoing challenge of handling enormous amounts of student data in biometric systems(Darapaneni et al., 2020). also Other majority of a biometrics application proposal focuses on the tool that the instructor will utilize in class. Every time the lecturer lets a late student into the class, this will eventually lead to disruptions and distractions. It will also take longer if biometrics are used because students will have to wait in line to use the device(Elias et al., 2019). To cub this issues a face recognition system is the ideal solution to required. Students' faces will be captured in real time as they are learning in class using face detection, ensuring a seamless learning experience. In order to sign for their attendance and retain some of the material that the lecturer has provided, students can also pay close attention without being bothered. For the lecturer, keeping track of every student's attendance register for reports at a later time is hassle-free because the system generates the attendance automatically. Because there is no longer any possibility for students to falsify the documents, the data generated for the report will be accurately and directly updated in a database system(Elias et al., 2019).
This project leverage deep learning for the management of attendance and emotion, it intends to add to the expanding corpus of research and innovation in the field of educational technology by creating an integrated system that combines emotion detection and attendance monitoring. The results of this project will ultimately improve educational outcomes and student well-being by assisting educators and educational institutions in better understanding and addressing the behavioral and emotional aspects of student learning. 

OBJECTIVES:
i.	To obtain and preprocess emotion dataset.
ii.	To build and train emotion recognition system based on the collected data.
iii.	To development attendance and emotion management system based on the model in ii.
iv.	To integrate sub-systems developed into a fully functional system.
v.	To evaluate the system performance using accuracy, precision, recall and Response time.
vi.	
OPERATIONAL STEPS:
Setting up Raspberry Pi.
1.	acquire a Raspberry Pi 3 or earlier alongside a Raspberry Pi camera and a 32GB SD card (available on Microscale)
2.	carefully and firmly connect the Raspberry Pi camera to the Raspberry Pi port.
3.	On your laptop, install Raspberry Pi Imager from the link Raspberry Pi OS – Raspberry Pi.
4.	Insert the SD card into the macro SD card and insert the macro SD card into your laptop.
5.	Using the imager, install Raspbian OS on the SD card, then carefully remove and insert into the Raspberry Pi (guide in the YouTube video: How To Install & Set Up Raspberry Pi OS - Pi4 Pi3 Pi2
6.	Connect the keyboard and mouse to the USB ports of the raspberry and monitor to the hdmi port either using direct hdmi or a VGA to hdmi converter.
7.	Connect power to the pi, and it will power up. Set up the user name and password and other required details.
8.	Update the pi and upgrade by running the following lines of code command on the command prompt one at a time:
sudo apt update
Sudo apt upgrade
9.	Install libacam on the raspberry pi following the below instruction:
i.	Run this command sudo raspi-config
ii.	Navigate to Interface Options > Camera and enable it.
iii.	Reboot the pi by running sudo reboot
iv.	Run the command sudo apt install -y libcamera-apps to install libcam
9.	Open up Thorny by clicking on the Raspberry Pi icon and then the programming and thorny
1.	Create a new Python file in Thorny and paste the raspberry pi code from github (AgabaEmbedded/Classroom-Assistant-System).
10.	Save the file and exit.
11.	Turn on the hotspot on your laptop and connect the Raspberry Pi to it. Set it to connect automatically.
12.	Set the Python script to run on startup. To do this, follow these steps: 
i.	Run the command sudo nano /etc/rc.local on the command prompt.
ii.	Inset the line python3 /path to your_script.py & before exit 0 and replace path to your_scripts.py with the path to your Python file created.
iii.	Save the file and close.


Setting up Classroom Assistant software.
1.	Download and install Python 3.12.0 and earlier version from Download Python | Python.org, and ensure to check the ‘add Python to directory’ box when installing Python.
2.	Download the software from Google Drive. Classroom assistant system: Google Drive
3.	Extract the software and move to a directory of choice. Note: The directory cannot be changed after the software is set up.
4.	Launch the command prompt using (windows + R) and then type cmd and hit enter.
5.	Navigate to the directory where the software is located using cd directory_of_the_software. Replace directory_of_the_software with the directory of the attendance software.
6.	Run the following command with a good internet connection.
pip install --upgrade pip
pip install virtualenv
virtualenv venv
venv\scripts\activate
pip install streamlit streamlit_option_menu deepface matplotlib tf-keras
7.	Turn on hotspot on your laptop and with the password and username used when configuring the raspberry pi that way the pi connects automaticly.
8.	Navigate to hotspot setting on your laptop and get the IP address of the raspberry pi, open the script app.py  in the software directory using vscode or any python development environment available and edit the variable raspi_ip in line 24 to the Ip address of the raspberry pi gotten. Note this step is only required once on the first run.


Using the Classroom Assistant Software
1.	Now software is ready to launch. Turn on wifi on Hotspot on your laptop using the password and username set when setting up the Raspberry Pi.
2.	Click on the classroom assistant file in the software directory to launch the software (do this every time to start the software).
3.	On start-up, a default course is created, “default course." You can create new courses by clicking “create course” in the homepage after entering the course title.
4.	Enrolling a student is required before taking attendance and entering exam mode; navigate to the “Enroll Student” page to enroll a new student. Select the course and enter the student’s details as required, then click on the ‘Enroll student’ button. If no image is uploaded, the laptop camera will capture the face in front of the laptop and extract the face from it as the passport of the student. Ensure there is a face in front of the laptop camera if you’re using this option in order not to run into an error.
5.	To delist a student from the attendance list, navigate to the “Remove Student” page, enter the required details of the student, and click Remove student.
6.	The Start Class page handles the attendance and emotion management. To commence class, select the course, enter the intended duration of the class in minutes, and click the start class button. 
a.	Note: good internet connection is required for the first class for the software to download needed models.
7.	To end an ongoing class abruptly, click on the End class button, which shows when the class is on.
8.	Exam mode manages exams. To use this feature, select the course title and the intended duration of the examination and click the commence exam button.
9.	Attendance can be downloaded in Excel format from the homepage; simply enter the course and click Download Attendance.
10.	The downloaded attendance contains the list of registered students and columns representing the date and start time of all the classes ever conducted as column headers, while the content of the column is the time that each student left the class or time the class ended if the student stayed till the end of the class or ‘0’ for students who were completely absent from the class.
11.	While using this software, if you encounter any issue, refer to the github repo: https://github.com/AgabaEmbedded/Classroom-Assistant-System.
12.	Contact sundayabraham81@gmail.com for more information and collaboration.



SUPERVISED BY:		Engr. Dr. I. M. Abdullahi
SUBMITTED TO:		Engr. B.K Nuhu

