import tkinter as tk
from tkinter import ttk
import requests
from PIL import Image
from io import BytesIO
import  matplotlib.pyplot as plt
import numpy as np


# Тестовый маршрут : инициализирует  окно для ввода  url, path , делает postrequest  к post маршруту

def out_form():
    root = tk.Tk() # Создаем главное окно
    root.title("Введите адрес документа")
    root.geometry("600x200")  # Устанавливаем размер окна (ширина x высота)

    label = tk.Label(root, text="Введите адрес документа:")
    label.pack(pady=10)

    entry = tk.Text(root, width=70, height=3)  # Ширина поля ввода
    entry.pack(pady=10)
    entry.insert(tk.INSERT, "data\\405___e989c7d20dd04eec89042272ca1a84b3_3.png") 
    
    button_frame = tk.Frame(root) # фрейм для кнопок
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    cancel= True 
    def cancel_ok():
        nonlocal cancel
        cancel=False
        root.quit()


    # Добавляем кнопки
    button_ok = tk.Button(button_frame, text="OK",width=20, command=root.quit)
    button_ok.pack(side=tk.LEFT, padx=20, pady=20)
    button_cancel = tk.Button(button_frame, text="Cancel",width=20, command=cancel_ok)  # Закрываем окно при нажатии "Отмена"
    button_cancel.pack(side=tk.RIGHT, padx=20, pady=20)

    root.protocol("WM_DELETE_WINDOW", cancel_ok)

    root.mainloop() # Запускаем главный цикл

    path = entry.get("1.0", tk.END).strip()
    root.destroy()

  
    return  path, cancel


def make_request(url_base, data):
    
    root = tk.Tk()
    label = tk.Label(root, text="Executing request..")
    label.pack()
    root.update() 

    response = requests.post(url_base, json=data)
    root.destroy()
    return response


def image_show(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
        
    plt.figure(figsize=(4,3))
    plt.imshow(image) 
    
    plt.xticks([]) 
    plt.yticks([])
    plt.xlabel(f'размер:{np.array(image).shape}')
    plt.show()
    

# Пример вызова функции
# make_request_and_close(url_base, data)


cancel='True'
while cancel:
    path ,cancel= out_form()
    if not cancel:
        break

    url_base='http://127.0.0.1:8000/forms/'
    data = {"path": path}
    # response =requests.post(url_base, json=data)
    response=make_request(url_base,data)
    print(response.json())
    image_show(response.json().get('save_url'))
   