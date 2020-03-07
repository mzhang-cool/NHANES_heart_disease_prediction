import tkinter as tk
import call_model as cm
from tkinter import ttk

window = tk.Tk()
window.title("SCOR Team07")
window.geometry("701x701")
canvas = tk.Canvas(window, height = 700, width = 700, bg = "#263D42")
canvas.place(relx = 0, rely = 0)
my_row = list(range(15,690,40))
c1 = 15
c2 = 250
c3 = 350
c4 = 440
#first row
frame = tk.Frame(window, bg = '#ECECEC')
frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)
frame_btn = tk.Frame(window, bg = "#263D42")
frame_btn.place(relwidth = 1, relheight = 0.1, rely = 0.9)

lbl = tk.Label(window, text="Scor", font=("Arial Bold", 30), fg = 'white', bg = "#263D42")
lbl_gender = tk.Label(frame, text = "Gender: ", font=("Arial", 20), bg = '#ECECEC')
lbl.place(x = 300, y = 5)
lbl_gender.place(x= c1, y = my_row[0])

selected = tk.IntVar()

rad1 = tk.Radiobutton(frame, text='Male', value=1, variable=selected, font=("Arial", 18), bg = '#ECECEC')
rad2 = tk.Radiobutton(frame, text='Female', value=2, variable=selected, font=("Arial", 18), bg = '#ECECEC')
rad1.place(x = 90, y =  my_row[0])
rad2.place(x = 180, y =  my_row[0])

lbl_age = tk.Label(frame, text = 'Age:', font=("Arial", 20), bg = '#ECECEC')
lbl_age.place(x = c3, y = my_row[0])
entry_age = tk.Entry(frame, width = 5)
entry_age.place(x = c4, y = my_row[0])
#second row

lbl_h = tk.Label(frame, text = "Height: ", font=("Arial", 20), bg = '#ECECEC')
lbl_h.place(x= c1, y =  my_row[1])

entry_h = tk.Entry(frame, width = 5)
entry_h.place(x = 85, y =  my_row[1])
lbl_kg = tk.Label(frame, text = 'cm', font=("Arial", 20), bg = '#ECECEC')
lbl_w = tk.Label(frame, text = 'Weight:              kg', font=("Arial", 20), bg = '#ECECEC')
lbl_w.place(x = c3, y = my_row[1])
lbl_kg.place(x = 150, y = my_row[1])
entry_w = tk.Entry(frame, width = 5)
entry_w.place(x = c4, y = my_row[1])

#third row
lbl_ly = tk.Label(frame, text = 'Lymphocyte:',font=("Arial", 20), bg = '#ECECEC')
lbl_ly.place(x = c1, y = my_row[2])
entry_ly = tk.Entry(frame, width = 5)
entry_ly.place(x = 150, y = my_row[2])

lbl_alb = tk.Label(frame, text = 'Albumin:',font=("Arial", 20), bg = '#ECECEC')
lbl_alb.place(x = c3, y = my_row[2])
entry_alb = tk.Entry(frame, width = 5)
entry_alb.place(x = c4, y = my_row[2])

#forth row
lbl_LBXNEPCT = tk.Label(frame, text = 'Segmented neutrophils:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXNEPCT.place(x = c1, y = my_row[3])
entry_LBXNEPCT = tk.Entry(frame, width = 5)
entry_LBXNEPCT.place(x = c2, y = my_row[3])

lbl_LBXPLTSI = tk.Label(frame, text = 'Platelet:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXPLTSI.place(x = 350, y = my_row[3])
entry_LBXPLTSI = tk.Entry(frame, width = 5)
entry_LBXPLTSI.place(x = 440, y = my_row[3])

#fifth row
lbl_LBXMCHSI = tk.Label(frame, text = 'Mean cell hemoglobin:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXMCHSI.place(x = c1, y = my_row[4])
entry_LBXMCHSI = tk.Entry(frame, width = 5)
entry_LBXMCHSI.place(x = c2, y = my_row[4])

lbl_LBXBAPCT = tk.Label(frame, text = 'Basophil:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXBAPCT.place(x = 350, y = my_row[4])
entry_LBXBAPCT = tk.Entry(frame, width = 5)
entry_LBXBAPCT.place(x = 440, y = my_row[4])

#sixth row
lbl_LBXMOPCT = tk.Label(frame, text = 'Monocyte percent:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXMOPCT.place(x = c1, y = my_row[5])
entry_LBXMOPCT = tk.Entry(frame, width = 5)
entry_LBXMOPCT.place(x = c2, y = my_row[5])

#seventh row
lbl_LBXRDW = tk.Label(frame, text = 'Red cell distribution width:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXRDW.place(x = c1, y = my_row[6])
entry_LBXRDW = tk.Entry(frame, width = 5)
entry_LBXRDW.place(x = c2+5, y = my_row[6])

#eighth row
lbl_LBXWBCSI = tk.Label(frame, text = 'White blood cell count:',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXWBCSI.place(x = c1, y = my_row[7])
entry_LBXWBCSI = tk.Entry(frame, width = 5)
entry_LBXWBCSI.place(x = c2, y = my_row[7])

#ninth row
lbl_LBXEOPCT = tk.Label(frame, text = 'Eosinophils percent :',font=("Arial", 20), bg = '#ECECEC')
lbl_LBXEOPCT.place(x = c1, y = my_row[8])
entry_LBXEOPCT = tk.Entry(frame, width = 5)
entry_LBXEOPCT.place(x = c2, y = my_row[8])
#tenth row
lbl_edu = tk.Label(frame, text = "Your highest education level:",font=("Arial", 20), bg = '#ECECEC')
lbl_edu.place(x = c1, y = my_row[9])
comboEdu = ttk.Combobox(frame, values=["Less Than 9th Grade", "9-11th Grade",
                                           "High School Grad/GED or Equivalent", "Some College or AA degree",
                                           "College Graduate or above"], width = 50)

comboEdu.place(x = c1, y = my_row[10])
comboEdu.current(1)

print(comboEdu.current(), comboEdu.get())
def clicked():
    #'LBXPLTSI','BMXHT','LBXBAPCT', 'BMXBMI','LBXMOPCT','RIDAGEYR', 'LBXMCHSI','LBXRDW','URXUMA',
    # 'LBXWBCSI','LBXEOPCT','LBXLYPCT','LBXNEPCT',RIAGENDR，DMDHREDU
    # 'BMXBMI' kg/m**2，DMDHREDU

    my_dict = {"Less Than 9th Grade": 1, "9-11th Grade": 2, "High School Grad/GED or Equivalent": 3,
               "Some College or AA degree": 4, "College Graduate or above": 5}
    w = float(entry_w.get())
    BMXHT = float(entry_h.get())
    h = BMXHT/100
    BMXBMI = w/(h*h)
    URXUMA = float(entry_alb.get())
    RIDAGEYR = float(entry_age.get())
    RIAGENDR = float(selected.get())-1
    LBXLYPCT = float(entry_ly.get())
    LBXPLTSI = float(entry_LBXPLTSI.get())
    LBXNEPCT = float(entry_LBXNEPCT.get())
    LBXBAPCT = float(entry_LBXBAPCT.get())
    LBXRDW = float(entry_LBXRDW.get())
    LBXMCHSI = float(entry_LBXMCHSI.get())
    LBXEOPCT = float(entry_LBXEOPCT.get())
    LBXMOPCT = float(entry_LBXMOPCT.get())
    LBXWBCSI = float(entry_LBXWBCSI.get())
    DMDHREDU = float(my_dict.get(comboEdu.get()))
    x_raw ={'BMXHT': BMXHT, 'BMXBMI': BMXBMI, 'URXUMA': URXUMA, 'RIDAGEYR':RIDAGEYR,
             'LBXLYPCT':LBXLYPCT, 'LBXPLTSI':LBXPLTSI,'LBXNEPCT':LBXNEPCT,
            'LBXBAPCT': LBXBAPCT, 'LBXRDW':LBXRDW,'LBXMCHSI':LBXMCHSI, 'LBXEOPCT':LBXEOPCT,
            'LBXMOPCT':LBXMOPCT,'LBXWBCSI':LBXWBCSI, 'DMDHREDU':DMDHREDU,'RIAGENDR_2.0': RIAGENDR}


    print(BMXBMI)
    print(DMDHREDU)
    print('platelet:')
    print(LBXPLTSI)
    cm.CallModel(x_raw)


btn = tk.Button(frame_btn, text="Submit", command=clicked, padx = 10, pady = 5, fg = 'black', bg = "#263D42")
btn.place(x = 300, y = 10)

window.mainloop()
