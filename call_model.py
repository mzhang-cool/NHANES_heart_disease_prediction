import pickle
import pandas as pd
import tkinter as tk
from tkinter import ttk
def CallModel(data):
    # load the model from disk
    df = pd.DataFrame(data,index=[0])
    my_model = "/Users/zhangmeng/Desktop/study/scor/repo_team07/finalized_model.sav"
    loaded_model = pickle.load(open(my_model, 'rb'))
    y = loaded_model.predict(df)
    print(y)
    printResult(y)
    print(y)


def printResult(result):
    popup = tk.Tk()
    popup.geometry("200x100")
    popup.wm_title("Result of heart disease potential")
    label = tk.Label(popup, text=result, font=("Verdana", 12), bg = "#ECECEC")
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="OK", command=popup.destroy)
    B1.pack()
    popup.mainloop()
