import tkinter as tk
from tkinter import ttk
import requests
from PIL import Image
from io import BytesIO
import  matplotlib.pyplot as plt
import numpy as np


# Тестовый маршрут : инициализирует  окно для ввода  url, path , делает post request  
# Распечатывает документ



# функция создает окно для ввода path  или url,  возвращает path : str ,  cancel :  bool для обработки закрытия окна
def out_form():
    root = tk.Tk() # Создаем главное окно
    root.title("Введите адрес документа")
    root.geometry("600x200")  # Устанавливаем размер окна (ширина x высота)

    label = tk.Label(root, text="Введите адрес документа:")
    label.pack(pady=10)

    entry = tk.Text(root, width=70, height=3)  # Ширина поля ввода
    entry.pack(pady=10)
    entry.insert(tk.INSERT, 'input_images\\405___87bdaaa7ba5247c1969398f16a341917_1.png') 
    
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

#  post request +  индикатор загрузки
def make_request(url_base, data):
    
    root = tk.Tk()
    label = tk.Label(root, text="Executing request..")
    label.pack()
    root.update() 

    response = requests.post(url_base, json=data)
    root.destroy()
    return response

# Тестовая печать документа
def image_show(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
        
    plt.figure(figsize=(8,6))
    plt.imshow(image) 
    
    plt.xticks([]) 
    plt.yticks([])
    plt.xlabel(f'размер:{np.array(image).shape}')
    plt.show()
    



cancel='True'
while cancel:
    path ,cancel= out_form()
    if not cancel:
        break
    print (path)
    url_base='http://127.0.0.1:8000/forms/'
    data = {"path": path}
    
    response=make_request(url_base,data)
    print(response.json())
    
    if response.json().get('save_url'):
        image_show(response.json().get('save_url'))
   