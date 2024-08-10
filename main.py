import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import tkinter as tk
import threading
import simpleaudio as sa

from playsound import playsound
from tkinter import *
from tkinter import Tk , Label
from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import VideoFileClip

############################################################################
#################        Author: Pham Thanh Tri       ######################
####################    Topic: Face Recognition   ##########################
############################################################################
def update(ind):
    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0
    label.configure(image=frame)
    root.after(100, update, ind)

patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    ' ': '-',
    '[ỳýỷỹỵ]': 'y',
}

def convert(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

predictor_path = "shape_predictor_68_face_landmarks.dat"  # Adjust this path if the file is located elsewhere
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def extract_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])

    def crop_feature(points, padding=10):
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        x_min = max(min(x_coords) - padding, 0)
        y_min = max(min(y_coords) - padding, 0)
        x_max = min(max(x_coords) + padding, image.shape[1])
        y_max = min(max(y_coords) + padding, image.shape[0])
        return gray[y_min:y_max, x_min:x_max]

    left_eye = crop_feature([landmarks.part(i) for i in range(36, 42)])
    right_eye = crop_feature([landmarks.part(i) for i in range(42, 48)])
    nose = crop_feature([landmarks.part(i) for i in range(27, 36)])
    mouth = crop_feature([landmarks.part(i) for i in range(48, 68)])
    return left_eye, right_eye, nose, mouth

def dft_magnitude(image):
    image = np.float32(image)
    fourier = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude = cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1])
    magnitude = 20 * np.log(magnitude + 1)  # Adding 1 to avoid log(0) and improve stability
    return magnitude

def resize_image(image, size):
    return cv2.resize(image, size)

def compare_features(feature1, feature2):
    # Resize both images to the same size
    feature1_resized = resize_image(feature1, (100, 100))
    feature2_resized = resize_image(feature2, (100, 100))

    dft_feature1 = dft_magnitude(feature1_resized)
    dft_feature2 = dft_magnitude(feature2_resized)
    
    # Normalize the magnitude images to the range [0, 1]
    dft_feature1 = cv2.normalize(dft_feature1, None, 0, 1, cv2.NORM_MINMAX)
    dft_feature2 = cv2.normalize(dft_feature2, None, 0, 1, cv2.NORM_MINMAX)

    # Compute SSIM
    ssim_similarity = ssim(dft_feature1, dft_feature2, data_range=dft_feature1.max() - dft_feature1.min())

    return ssim_similarity

def find_and_compare_faces(sample_image, target_image):
    sample_features = extract_facial_features(sample_image)
    target_features = extract_facial_features(target_image)
    
    if sample_features is None or target_features is None:
        return 0, 0, 0, 0  # Return 0 similarity if no face is detected in either image

    left_eye_similarity = compare_features(sample_features[0], target_features[0])
    right_eye_similarity = compare_features(sample_features[1], target_features[1])
    nose_similarity = compare_features(sample_features[2], target_features[2])
    mouth_similarity = compare_features(sample_features[3], target_features[3])

    return left_eye_similarity, right_eye_similarity, nose_similarity, mouth_similarity

images_path = None
filename = None
check = None

def RunCode():
    global filename, images_path, check
    audio()
    folder_dir = images_path
    
    sample_image_main = cv2.imread(filename)
    if sample_image_main is None or folder_dir is None:
        help = messagebox.askokcancel('Error','Please select photos and folders before clicking Ok')
    else: 
        threshold = 0.30
        lowshold = 0.25
        ###
        # LabelSimilar()
        files = os.listdir(folder_dir)
        for file in files:
            link=os.path.join(folder_dir, file)
            newlink= os.path.join(folder_dir, convert(file))
            oldpath = link    
            newpath = newlink
            os.rename(oldpath, newpath)
        ###
        check = False
        for image_name in files:
            if image_name.endswith((".png", ".jpg", ".jpeg")):
                '''  
                link=os.path.join(folder_path, file)
                '''
                image_path= os.path.join(folder_dir, image_name) 
                image = cv2.imread(image_path)   
                
                left_eye_similarity, right_eye_similarity, nose_similarity, mouth_similarity = find_and_compare_faces(sample_image_main, image)
            similarities_below_threshold = [
                left_eye_similarity < lowshold,
                right_eye_similarity < lowshold,
                nose_similarity < lowshold,
                mouth_similarity < lowshold
            ]
            similarities_above_threshold = [
                left_eye_similarity > threshold,
                right_eye_similarity > threshold,
                nose_similarity > threshold,
                mouth_similarity > threshold
            ]
            average_similarity = (left_eye_similarity + right_eye_similarity + nose_similarity + mouth_similarity) / 4
            if (sum(similarities_above_threshold) >= 3 or average_similarity > 0.29) and sum(similarities_below_threshold) < 1:
                    check = True
                    complete()       
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].imshow(cv2.cvtColor(sample_image_main, cv2.COLOR_BGR2RGB))
                    ax[0].set_title('Input Image')
                    ax[0].axis('off')
                    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax[1].set_title('Similar Face')
                    ax[1].axis('off')
                    similarity_text = (f'Left Eye Similarity: {left_eye_similarity}\n'
                                    f'Right Eye Similarity: {right_eye_similarity}\n'
                                    f'Nose Similarity: {nose_similarity}\n'
                                    f'Mouth Similarity: {mouth_similarity}')

                    plt.figtext(0.5, 0.01, similarity_text, ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

                    plt.show()
        if check is True:
            response=messagebox.askyesno('Finish!','Find successful')
        else:
           response=messagebox.askyesno('Not found!','Not found similar images')
        
def complete():
    threadsound = threading.Thread(target=playsound('complete.mp3'))
    threadsound.start()

def DestroyFile():
    global Delete, my_image_label, filename
    audio()
    my_image_label.destroy()
    Delete.place_forget() 
    filename = None

def DestroyFolder():
    global DeleteFolder, my_folder_label, images_path
    audio()
    my_folder_label.destroy()
    DeleteFolder.place_forget() 
    images_path = None

def play_sound():
    playsound('click.mp3')

def audio():
    threadsound = threading.Thread(target=play_sound)
    threadsound.start()

def GetPathFolder():
    global images_path, DeleteFolder, my_folder_label
    audio()
    images_path = filedialog.askdirectory()

    if images_path != "":
        folder = 'folder.jpg'
        my_folder = Image.open(folder)
        my_folder.thumbnail(size=[300,250])
        tkimg = ImageTk.PhotoImage(my_folder)
        my_folder_label = tk.Label(root, width = 200, height= 150)
        my_folder_label.place(x = 0, y =0)
        my_folder_label.config(image=tkimg)
        my_folder_label.image = tkimg
        DeleteFolder = Button(root, text="Delete", bg='red', fg='black', command=lambda:DestroyFolder() )
        DeleteFolder.place(x = 0, y = 0, width= 80, height = 30)
        return images_path

def GetPathFile():
    audio()
    global filename, my_image, Delete, my_image_label
    filename = filedialog.askopenfilename( filetypes=(("jpg files" , "*.jpg"),("png files", "*.png"),("all files", "*.*")))
    if filename != "":
        my_label.config(text=filename)
        my_image = Image.open(filename)
        my_image.thumbnail(size=[300,250])
        tkimg = ImageTk.PhotoImage(my_image)
        my_image_label = tk.Label(root, width = 200, height= 150)
        my_image_label.place(x = 600, y =0)
        my_image_label.config(image=tkimg)
        my_image_label.image = tkimg
        Delete = Button(root, text="Delete", bg='red', fg='black', command=lambda:DestroyFile() )
        Delete.place(x = 720, y = 0, width= 80, height = 30)
        return filename   

def ButtonInterface():
    canvas.create_text(400, 50, text="WELCOME TO FACE RECOGNITION!", font="calibri 20 bold", fill="white")

def ButtonOk():
    Ok = Button(root, text ='Ok', bg= 'yellow',command=RunCode).place(x = 380, y = 305, width= 40, height= 30)

def ButtonFolder():
    folder = Button(root,text ="Open Folder",bg='yellow', command=GetPathFolder).place(x = 300, y = 290, width=80, height = 60)

def ButtonFile():
    file = Button(root, text='Open Images',bg='yellow', command=GetPathFile).place(x = 420, y= 290, width = 80, height = 60)

def ButtonExit():
    Exit = Button(root, text = "Exit",fg= 'black', bg= 'red',command=on_close).place(x = 750, y = 575, width=50 , height= 25)

def ButtonHelp():
    help = Button(root, text = 'Help', command=help_content, bg = 'blue',fg = 'black').place(x = 0, y=575, width= 50, height=25)

def on_close():
    audio()
    response=messagebox.askyesno('Exit','Are you sure you want to exit?')
    if response:
        audio()
        root.destroy()
    else:
        audio()

def help_content():
    audio()
    help = messagebox.askokcancel('Help','Open your folder and up a chosen folder, similarity with Open File. Finally click Ok.')
    audio()

#############################################################

# for the filename of selected image
if __name__ == "__main__":
    root = Tk()
    pathlogo = "logo.jpg"
    load_logo = Image.open(pathlogo)
    render_logo = ImageTk.PhotoImage(load_logo)
    root.iconphoto(False, render_logo)
    root.wm_title("Face Recognition")
    root.minsize(width=800, height=600)
    root.maxsize(width=800, height=600)
    my_label = tk.Label(root)
    my_label.pack()
    canvas= Canvas(root,width= 800, height= 600)
    canvas.pack(expand=True, fill= BOTH)
    frameCnt = 12
    frames = [PhotoImage(file='gifbk.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]
    label = Label(root, width= 800, height=600)
    label.place(x = 0, y = 0)
    root.after(0, update, 0)
    ######
    ButtonOk()
    ButtonFile()
    ButtonFolder()
    ButtonHelp()
    ButtonInterface()
    ButtonExit()
    root.mainloop()  
    exit(0)


