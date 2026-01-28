# REVOLUTINIZING-HEALTHCARE-AI-POWERED-BONE-FRACTURE-DETECTION-SYSTEM

### INTRODUCTION

This project focuses on the task of automating and enhancing the accuracy of bone fracture classification in X-ray images. In the past, this diagnosis has predominantly depended on the skills of radiologists, but issues such as tiredness, heavy workloads, and differences in how individuals interpret information can result in inconsistencies. To tackle this issue, we suggest a system that utilizes Convolutional Neural Networks (CNNs), which are a kind of machine learning specifically skilled at identifying images. This study explores how pre-trained CNN models, specifically ResNet50, can be used to detect and categorize fractures in three different bone areas: elbow, hand, and shoulder. We make use of the MURA dataset, which consists of more than 20,335 musculoskeletal radiographs that are publicly available, in order to train and validate the suggested method.
The main concept of the system revolves around a two-step process to accurately classify fractures. In the beginning phase, ResNet50, well-known for its strong image classification capabilities, analyzes the X-ray image to identify the specific type of bone shown. The initial step serves as a crucial filter in focusing the system on the accurate bone structure and potential fracture types for future evaluation. Once the bone type is identified, the system proceeds to the subsequent step. In this scenario, a custom-designed model is used to detect a break in the elbow, hand, or shoulder following specific training for that particular bone.

### SOFTWARE ENVIRONMENT

*	Operating system	: Windows7 (with service pack 1), 8, 8.1 ,10 and 11
*	Language	: Python
*	Libraries & Frameworks: TensorFlow / Keras, PyTorch OpenCV, NumPy, Pandas Matplotlib, Seaborn
CUDA & cuDNN, TensorBoard
*	Dataset & Storage:
MURA Dataset
Google Drive / AWS S3 / Local Storage
*	Development Tools:
Jupyter Notebook / Google Colab / PyCharm / VS Code

### PART OF CODE
~~~
import os

from tkinter import filedialog import customtkinter as ctk import pyautogui
import pygetwindow

from PIL import ImageTk, Image from predictions import predict
project_folder = os.path.dirname(os.path.abspath( file )) folder_path = project_folder + '/images/'
filename = ""
class App(ctk.CTk): def   init  (self):
super().  init  ()

self.title("Bone Fracture Detection") self.geometry(f"{500}x{740}") self.head_frame = ctk.CTkFrame(master=self)
self.head_frame.pack(pady=20, padx=60, fill="both", expand=True) self.main_frame = ctk.CTkFrame(master=self) self.main_frame.pack(pady=20, padx=60, fill="both", expand=True)
self.head_label = ctk.CTkLabel(master=self.head_frame, text="Bone Fracture Detection", font=(ctk.CTkFont("Roboto"), 28))
self.head_label.pack(pady=20, padx=10, anchor="nw", side="left") img1 = ctk.CTkImage(Image.open(folder_path + "info.png"))
self.img_label	=	ctk.CTkButton(master=self.head_frame,	text="", image=img1, command=self.open_image_window,width=40,
height=40)

self.img_label.pack(pady=10, padx=10, anchor="nw", side="right")

self.info_label = ctk.CTkLabel(master=self.main_frame, text="Bone fracture detection system, upload an x-ray image for fracture detection.", wraplength=300, font=(ctk.CTkFont("Roboto"), 18))

self.info_label.pack(pady=10, padx=10)

self.upload_btn	=	ctk.CTkButton(master=self.main_frame,	text="Upload Image", command=self.upload_image)

self.upload_btn.pack(pady=0, padx=1)

self.frame2 = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=256, height=256)

self.frame2.pack(pady=10, padx=1)

img = Image.open(folder_path + "Question_Mark.jpg")
img_resized = img.resize((int(256 / img.height * img.width), 256)) # new width & height img = ImageTk.PhotoImage(img_resized)
self.img_label = ctk.CTkLabel(master=self.frame2, text="", image=img) self.img_label.pack(pady=1, padx=10)
self.predict_btn=ctk.CTkButton(master=self.main_frame,text="Predict",command=self.predict_gui) self.predict_btn.pack(pady=0, padx=1)

self.result_frame = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=200, height=100)

self.result_frame.pack(pady=5, padx=5)

self.loader_label = ctk.CTkLabel(master=self.main_frame, width=100, height=100, text="") self.loader_label.pack(pady=3, padx=3)
self.res1_label = ctk.CTkLabel(master=self.result_frame, text="") self.res1_label.pack(pady=5, padx=20)
self.res2_label = ctk.CTkLabel(master=self.result_frame, text="")

self.res2_label.pack(pady=5, padx=20)

self.save_btn=ctk.CTkButton(master=self.result_frame,text="SaveResult",command=self.save_result) self.save_label = ctk.CTkLabel(master=self.result_frame,
text="") def upload_image(self): global filename
f_types = [("All Files", "*.*")]

filename = filedialog.askopenfilename(filetypes=f_types, initialdir=project_folder+'/test/Wrist/') self.save_label.configure(text="")
self.res2_label.configure(text="") self.res1_label.configure(text="") self.img_label.configure(self.frame2, text="", image="") img = Image.open(filename)
img_resized = img.resize((int(256 / img.height * img.width), 256)) # new width & height img = ImageTk.PhotoImage(img_resized) self.img_label.configure(self.frame2, image=img,
text="") self.img_label.image = img self.save_btn.pack_forget() self.save_label.pack_forget()
def predict_gui(self): global filename
bone_type_result = predict(filename) result = predict(filename, bone_type_result) print(result)

if result == 'fractured':

self.res2_label.configure(text_color="GREEN",text="Result:Normal", font=(ctk.CTkFont("Roboto"), 24))

bone_type_result = predict(filename, "Parts")

self.res1_label.configure(text="Type: " + bone_type_result, font=(ctk.CTkFont("Roboto"), 24)) print(bone_type_result)
self.save_btn.pack(pady=10, padx=1) self.save_label.pack(pady=5, padx=20)
def save_result(self):

~~~

### OUTPUT

### Output of Normal Shoulder
<img width="322" height="513" alt="image" src="https://github.com/user-attachments/assets/2e35b657-eb7f-4928-a7f3-15c087c444cc" />

## Output of Fractured Bone detection for Shoulder
<img width="321" height="510" alt="image" src="https://github.com/user-attachments/assets/5b11cd27-5a9c-45e6-bf54-5f85a856f973" />

### RESULT

The proposed two-stage CNN-based fracture classification system demonstrated strong performance in accurately identifying and detecting bone fractures from X-ray images. By leveraging transfer learning with the pre-trained ResNet50 model for initial bone type recognition, followed by specialized bone-specific classifiers, the system achieved improved precision and reduced misclassification compared to a single generalized model. Experimental evaluation on the MURA dataset showed high classification accuracy, better feature extraction, and faster convergence during training. The bone-specific models were particularly effective in capturing subtle fracture patterns unique to the elbow, hand, and shoulder, leading to enhanced sensitivity and reliability in diagnosis. Overall, the results confirm that the proposed approach provides a robust, automated, and consistent solution that can assist radiologists by minimizing diagnostic errors, reducing workload, and enabling quicker clinical decision-making.
