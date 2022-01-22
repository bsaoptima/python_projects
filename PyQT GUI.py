import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tkinter as tk
from tkinter import ttk

''' ^^^^^^IMPORTS^^^^^^ '''

LARGE_FONT = ("Verdana", 12)
MEDIUM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=MEDIUM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

class VectorField(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default='icon.ico')
        tk.Tk.wm_title(self, "Vector Fields")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command=lambda: popupmsg('Not supported just yet!'))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        filemenu.add_cascade(label="File", menu=filemenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, Visualizer2D, Visualizer3D):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="This is the start page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="View Vector Field in 2D", command=lambda: controller.show_frame(Visualizer2D))
        button.pack()

        button2 = ttk.Button(self, text="View Vector Field in 3D", command=lambda: controller.show_frame(Visualizer3D))
        button2.pack()


class Visualizer2D(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Visualize vector fields in 2D!', font = LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="Home Page", command=lambda: controller.show_frame(StartPage))
        button.pack()

        f = plt.figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))

        u = np.cos(-y)
        v = np.sin(x)

        a.quiver(x, y, u, v)


        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Visualizer3D(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Visualize vector fields in 3D!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="Homepage", command=lambda: controller.show_frame(StartPage))
        button.pack()

        f = plt.figure(figsize=(5,5), dpi=100)
        a = f.gca(projection='3d')
        x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.8))

        u = np.cos(-z)
        v = np.sin(x)
        w = np.exp((y))

        a.quiver(x, y, z, u, v, w, length=0.1, color = 'black')

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = VectorField()
app.geometry("1280x720")
app.mainloop()
