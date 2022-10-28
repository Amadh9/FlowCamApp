
from tkinter import *
from tkinter import filedialog
from pollen_classification_2 import classify, crop_boxes
import time
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

gui = Tk()
gui.geometry("530x400")
gui.title("FlowCam Image Analysis Assistant")

folderPath = StringVar()
label1 = Label(gui, text='')
label1.place(x=40, y=60)


def getFolderPath():
    folder_selected = filedialog.askdirectory(initialdir='C:/')
    folderPath.set(folder_selected)
    label1.config(text= folder_selected)

def update_btn():
    new_text = 'Processing ...'
    if folderPath is None:
        label2.config(text="Please select folder")
    else:
        label2.config(text = new_text)
    gui.update_idletasks()

def process_fc():
    start_time = time.time()
    folder = folderPath.get()
    if folder is not None:
        tosave_cropped = folder+ '/cropped/'
        if not os.path.exists(tosave_cropped):
            os.mkdir(tosave_cropped)
            crop_boxes(folder,tosave_cropped)
        else:
            if len(os.listdir(tosave_cropped)) < 1:
                crop_boxes(folder, tosave_cropped)

        if len(os.listdir(tosave_cropped)) > 1:
            classify(folder)


        end_time = time.time()
        elapsed= time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))
        label1.config(text="Time Elapsed:" + elapsed)
        label2.config(text="Done!")




def exit_fc():
    gui.destroy()


a = Label(gui, text="Folder")
a.grid(row =0,column = 0)
E = Entry(gui,textvariable=folderPath,width=40)
E.grid(row=0,column=1)
btnFind = Button(gui, text="Select Folder",command=getFolderPath)
btnFind.grid(row=0,column=2)
text = StringVar()
text = ''
label2 = Label(gui, text= text)
label2.place(x=240, y=200)
btnProcess = Button(gui, height= 2, text="Process", command=lambda :[update_btn(), process_fc()])
btnProcess.place(x= 220, y=100)

btnExit = Button(gui, height=2, text="Exit", command=exit_fc)
btnExit.place(x=225, y=300)

gui.mainloop()