import numpy as np
import cv2
import math
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import os

class Compressor:
    def __init__(self):
        # side of zoomed-in image to show on UI
        self.panel_size = 500

        # range of compress ratio (0% to 20% of the original image)
        self.scale_min_point = 0
        self.scale_max_point = 20
        self.defaut_scale_point = int((self.scale_max_point - self.scale_min_point)/2)

        # setup UI
        self.root = tk.Tk()
        self.root.geometry("1200x800")
        self.root.resizable(False, False)
        self.root.title('Compressor')
        

        # Create A Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH,expand=1)
        # Create Frame for X Scrollbar
        sec = tk.Frame(main_frame)
        sec.pack(fill=tk.X,side=tk.BOTTOM)
        # Create A Canvas
        self.my_canvas = tk.Canvas(main_frame)
        self.my_canvas.pack(side=tk.LEFT,fill=tk.BOTH,expand=1)
        # Add A Scrollbars to Canvas
        x_scrollbar = ttk.Scrollbar(sec,orient=tk.HORIZONTAL,command=self.my_canvas.xview)
        x_scrollbar.pack(side=tk.BOTTOM,fill=tk.X)
        y_scrollbar = ttk.Scrollbar(main_frame,orient=tk.VERTICAL,command=self.my_canvas.yview)
        y_scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        # Configure the canvas
        self.my_canvas.configure(xscrollcommand=x_scrollbar.set)
        self.my_canvas.configure(yscrollcommand=y_scrollbar.set)
        self.my_canvas.bind("<Configure>", self.onCanvasConfigure) 
        # Create Another Frame INSIDE the Canvas
        self.second_frame = tk.Frame(self.my_canvas)
        # Add that New Frame a Window In The Canvas
        self.my_canvas.create_window((0,0),window=self.second_frame, anchor="nw", tags="second_frame")

        # tk.Grid.rowconfigure(self.second_frame, 0, weight=1)
        tk.Grid.columnconfigure(self.second_frame, 0, weight=1)
        tk.Grid.columnconfigure(self.second_frame, 1, weight=1)

        var = tk.DoubleVar()
        self.scale = tk.Scale(self.second_frame, variable=var, resolution=0.01, from_=self.scale_min_point, to=self.scale_max_point, tickinterval=1, orient=tk.HORIZONTAL, length=1200, label='% of original image')
        self.scale['state'] =tk.DISABLED
        self.scale['takefocus'] = 0
        self.scale.grid(row=0,column=0,columnspan=2,pady=10,padx=10)

        self.open_button = tk.Button(self.second_frame, text='Open file')
        self.open_button.bind('<Button-1>', self.select_file)
        self.open_button.grid(row=1,column=0,columnspan=2,pady=10,padx=10)

        self.org_label = tk.Label(self.second_frame, text="original image")
        self.org_label.grid(row=2,column=0,pady=10,padx=10,)

        self.update_label = tk.Label(self.second_frame, text="processed image")
        self.update_label.grid(row=2,column=1,pady=10,padx=10)

        self.org_panel = tk.Label(self.second_frame)
        self.org_panel.grid(row=3,column=0,pady=10,padx=10)

        self.update_panel = tk.Label(self.second_frame)
        self.update_panel.grid(row=3,column=1,pady=10,padx=10)

        self.root.mainloop()

    def onCanvasConfigure(self, event):
        self.my_canvas.config(scrollregion= self.my_canvas.bbox(tk.ALL))
        self.my_canvas.itemconfig('second_frame', height=self.my_canvas.winfo_height(), width=self.my_canvas.winfo_width())

    # open image file from local machine
    def select_file(self, event):
        filetypes = [("image files", ".jpg .jpeg .png")]
        file = fd.askopenfilename(title='Choose an image',initialdir='~/',filetypes=filetypes)

        if file:
            if len(self.scale.bind()) == 0:
                self.scale.bind('<ButtonRelease-1>', self.compress)
            filepath = str(os.path.abspath(file))
            org_img = cv2.imread(filepath)
            # number of rows and columns
            (self.blue, self.green, self.red) = cv2.split(org_img)
            self.rows, self.cols = self.blue.shape

            # identify center point of the image
            self.center_row, self.center_column = int(self.rows/2), int(self.cols/2)
#---------------------------------------------------------------------------------------------------------------------------------

            # compute output dimensions for image
            self.dim = (self.cols, self.rows)
            max_row_col = max(self.rows, self.cols)
            if max_row_col > self.panel_size:
                self.dim = (int(self.panel_size*self.cols/max_row_col), int(self.panel_size*self.rows/max_row_col))
            
            # reside and display the image
            scaled_img = cv2.resize(org_img, self.dim)
            im = Image.fromarray(scaled_img)
            imgtk = ImageTk.PhotoImage(image=im)
            self.org_panel.igmtk = imgtk
            self.org_panel.configure(image=imgtk)

            # enable slider if necessary
            if self.scale['state'] == tk.DISABLED:
                self.scale['state'] = tk.NORMAL

            # reset slider to default state for new image
            self.prev_filter_size = self.defaut_scale_point
            self.scale.set(self.defaut_scale_point)
            self.compress(None)

    def compress(self, event):
        filter_size = self.scale.get()
        if event is not None and filter_size == self.prev_filter_size:
            return

        # compute filter dimensions
        h = int(math.sqrt(0.01*filter_size)*self.rows/2)
        w = int(math.sqrt(0.01*filter_size)*self.cols/2)

        dft_blue = cv2.dft(np.float32(self.blue),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_blue = np.fft.fftshift(dft_blue)
        dft_green = cv2.dft(np.float32(self.green),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_green = np.fft.fftshift(dft_green)
        dft_red = cv2.dft(np.float32(self.red),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_red = np.fft.fftshift(dft_red)

        # create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((self.rows, self.cols, 2),np.uint8)
        mask[self.center_row-h:self.center_row+h, self.center_column-w:self.center_column+w] = 1

        # apply mask and inverse DFT to 3 channels of the image
        fshift_blue = dft_shift_blue*mask
        f_ishift_blue = np.fft.ifftshift(fshift_blue)
        img_back_blue = cv2.idft(f_ishift_blue, flags=cv2.DFT_SCALE)
        img_back_blue = cv2.magnitude(img_back_blue[:,:,0],img_back_blue[:,:,1])

        fshift_green = dft_shift_green*mask
        f_ishift_green = np.fft.ifftshift(fshift_green)
        img_back_green = cv2.idft(f_ishift_green, flags=cv2.DFT_SCALE)
        img_back_green = cv2.magnitude(img_back_green[:,:,0],img_back_green[:,:,1])

        fshift_red = dft_shift_red*mask
        f_ishift_red = np.fft.ifftshift(fshift_red)
        img_back_red = cv2.idft(f_ishift_red, flags=cv2.DFT_SCALE)
        img_back_red = cv2.magnitude(img_back_red[:,:,0],img_back_red[:,:,1])

        # cast all channels to original type
        img_back_blue = np.uint8(img_back_blue)
        img_back_green = np.uint8(img_back_green)
        img_back_red = np.uint8(img_back_red)

        # merge 3 channels
        img_back = cv2.merge([img_back_blue, img_back_green, img_back_red])

        # resize and display processed image
        scaled_img_back = cv2.resize(img_back, self.dim)
        im = Image.fromarray(scaled_img_back)
        imgtk = ImageTk.PhotoImage(image=im)
        self.update_panel.imgtk = imgtk
        self.update_panel.configure(image=imgtk)

        self.prev_filter_size = filter_size


if __name__ == "__main__":
    c = Compressor()