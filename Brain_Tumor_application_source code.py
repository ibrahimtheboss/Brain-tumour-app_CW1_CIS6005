#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import itertools
from numpy import argmax



my_w = tk.Tk()
my_w.geometry("1000x600")  # Size of the window 
my_w.title('Brain Tumor Classification Application by Ibrahim hakam')
my_w.configure(background='#f1f2f6')
my_font1=('arial', 18, 'bold')
l1 = tk.Label(my_w,text='Brain Tumor Classification',width=30,font=my_font1,cursor="mouse")  
l1.grid(row=1,column=1,columnspan=4)
l1.place(x=150,y=50)
b1 = tk.Button(my_w, text='Upload Image Files', 
   width=20,command = lambda:upload_file())
b1.configure(background='#2ecc71', foreground='white',font=('arial',10,'bold'))
b1.grid(row=2,column=1,columnspan=4)
b1.place(x=750,y=90)




b1 = tk.Button(my_w, text='Classify The image', 
width=20,command = lambda:classify_file())
b1.configure(background='#e74c3c', foreground='white',font=('arial',10,'bold'))
b1.grid(row=3,column=1,columnspan=4)
b1.place(x=750,y=140)

 # Create text widget and specify size.
T = Text(my_w, height = 5, width = 42,font=('arial',12,'bold'),wrap = WORD)
    # Create label
l = Label(my_w, text = "Prediction Result:",highlightcolor="black")
l.config(font =("Sans Serif", 14))
l.pack()
T.pack()
    
l.place(x=650,y=200)
T.place(x=580,y=230)
def clear_textbox():
    T.delete(1.0, 'end')
def filepath_dec():
    global Filepath_1
    Filepath_1 = []*1
    
def upload_file():
    filepath_dec()
    
    if len(Filepath_1)>=0:
        # We will have global variable to access these
        # variable and change whenever needed

        f_types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png')]   # type of files to select 
        filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
        if len(filename)>0: 
            Filepath_1.append(filename)
            if len(Filepath_1) >1:
                Filepath_1.pop()
            
            for f in filename:

                img=Image.open(f) # read the image file

                img=img.resize((384,384)) # new width & height
                img=ImageTk.PhotoImage(img)
                e1 =tk.Label(my_w)
                e1.image = img
                e1.place(x=60,y=120)
                e1['image']=img # garbage collection 
                    
def on_close():
    filepath_dec()
    response=messagebox.askyesno('Exit','Are you sure you want to exit?')
    if response:
        Filepath_1.clear()
        my_w.destroy()
my_w.protocol('WM_DELETE_WINDOW',on_close)# ask before exiting

def classify_file():
    
    global d
    d=[]
    if len(Filepath_1)==0:
        messagebox.showinfo("Attention","Please Upload an MRI image")
    else:
        d.clear()
        filename_1 = list(itertools.chain(*Filepath_1))
        filedatap = filename_1[0]
        

        image = tf.keras.preprocessing.image.load_img(filedatap, color_mode='grayscale', 
        target_size= (128,128))

        image=np.array(image)
        d.append(image)
        if len(d)>1:
            d.pop()
        d = np.array(d)
        #messagebox.askokcancel("Question","Do you want to open this file?")
        d = d.reshape((d.shape[0], 128, 128, 1))
        d = d.astype('float')
        d= d/255.0
        d.flatten() 

        # loading model
        model = keras.models.load_model('model_18.h5')
        ind = 0
        pred_flat= model.predict(d)


        if argmax(pred_flat) == 0:
            ind = 0 
            Fact =("The model has predicted the MRI image has Brain Tumor of Glioma"+
                    ' with a Confidence of '+"{:.2f}".format((pred_flat[0][ind]*100))+'%')

        elif argmax(pred_flat) == 1:
            ind = 1 
            Fact =("The model has predicted the MRI image has Brain tumour of Meningioma"+ 
                    ' with a Confidence of '+"{:.2f}".format((pred_flat[0][ind]*100))+'%')

        elif argmax(pred_flat) == 2:
            ind = 2 
            Fact =("The model has predicted the MRI image has No Brain Tumor!! "+
                    ' with a Confidence of '+"{:.2f}".format((pred_flat[0][ind]*100))+'%')

        elif argmax(pred_flat) == 3:
            ind = 3 
            Fact =("The model has predicted the MRI image has Brain Tumor of Pituitary"+
                    ' with a Confidence of '+"{:.2f}".format((pred_flat[0][ind]*100))+'%')

        else :
            Fact ="The model has encountered an error"


        clear_textbox()
        T.insert('end', Fact)

    
Button(
    my_w,
    text='Clear',
    width=15,
    height=2,background='#7f8c8d',
    command=clear_textbox
    ).place(x=750,y=360)

l1 = tk.Label(my_w,text='This Software is Designed and Developed by st20215372 Ibrahim Hakam',
              width=80,font=('arial',8,),background='#f1f2f6')  
l1.grid(row=1,column=1,columnspan=15)
l1.place(x=230,y=580)   
   


       
my_w.mainloop()  # Keep the window open


# In[ ]:




