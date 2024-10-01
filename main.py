'''
Этот  FastAPI приложение которое предсказывает положение документа ((Вертикально- 0, 90'- 1, 180'- 2, 270'- 3)),
на сонове предсказания трансформирует ориентацию в вертикальное положение,
записывает трансформировнный файл в папку 'transformed_images',
возвращает json ответ, содержащий:
{'post_request':data.path,  //строку  запроса : url или  путь к файлу,  указанный в post request
 'save_path':save_path, - //путь к файлу
 'save_url':save_url,  - //адрес  файла
 'pred:pred.item() - //предсказание положения докомента
}
Папка /app/transformed_images  смонтирована по маршруту /transformed_images
 Маршруты:
- `/forms/` (POST) вызывает def model_inference(data: FilePath)  data = {"path": path} где path
строка c адресом или путь к файлу.
FilePath  - пользовательский  класс, проверяющий тип перменной path : str,
в скрипте предусмотрна обработка пременной path для исключения ошибок выполнения скрипта.
функция  для  извлечения имени файла из get.request or url - name_url 
структура:
model_inference_final(model, path): принимеает  model -  обученная модель, path - переменая path 
    load_image(path) возвращает  загруженный документ image  и имя файла file_name
       check_url (path)  вызывается приусловии path  не путь, обрабатывает get request(path)
                          возвращает в  случае успеха  response
       name_url(response) возвращает  file_name

 возвращает  pred, save_path, save_url 
 '''                    






from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn

import requests
import os
import re
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Определим пользовательский класс NeuralNetwork для инициализации  модели 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= models.resnet50(weights='DEFAULT')
        num_features =self.model.fc.in_features
        self.model.fc=nn.Linear(num_features, 4)

    def forward(self, x):
       
        logits = self.model(x)
        return logits
    
# сheck_url -функция используемая в Load_image для проверки url и сохраняет в случае успеха ответ в response.

def check_url(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            raise HTTPException(status_code=400, detail='Invalid URL: The server responded with an error.')
    except:
        raise HTTPException(status_code=400, detail='Invalid path or URL: The specified path does not exist.')        
        
    

# name_url - функция  извлечения имени файла из get.request or url
def name_url(response,extension):
    content_disposition = response.headers.get('Content-Disposition', None)
    if content_disposition:
        match=re.search(r'filename=["\']?([^"\']+)["\']?', content_disposition)
        if match: 
            return  match.group(1)

 # Извлечение  имени файла из URL -  убираем название хоста, удаляем запрещенные символы, добавляем extension 
    name =re.sub(r'https?://[^/]+', '', response.url)
    name =re.sub(r'[<>:"/\\|?*]', '_', name)
    if name =='': 
        name='image_empty'
                
    return   f'{name}.{extension}'

            
                
    
# Load_image - обрабатывает, полученную строку data_path, определяя  ее как путь  или адрес,загружает изображение (PIL Image).
# Обрабатывает ошибки, возвращает  image  или или вызывает исключение (raise).
# 
def load_image(data_path: str):
    # Предполагаем, что path - путь к файлу
    data_path_norm = data_path.replace("\\", "/")                              
    if os.path.exists(data_path_norm): # действительно является локальным path
        try:
            image = Image.open(data_path_norm)
            file_name=os.path.basename(data_path_norm)
            return image, file_name
        except Exception as e: 
            raise HTTPException(status_code=400, detail=f'Error opening image from path: {str(e)}')

    elif check_url(data_path): # path не  путь! возможно url?
        try:
            response = check_url(data_path)
            image = Image.open(BytesIO(response.content))
            file_name= name_url(response, image.format)
            return image, file_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Error opening image from URL: {str(e)}')

    else:
        raise HTTPException(status_code=400, detail='Invalid path: The specified path does not exist.')
    


# Загрузка обученной модели 
model = NeuralNetwork()
model = torch.load(os.path.join('models','model.pth'), weights_only=False, map_location='cpu')   

# Пайплайн трансформаций изображений для подготовки данных к модели.
# Применяет преобразование в оттенки серого (с 3 каналами) и изменение размера до 224x224
transform_test_valid =transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224))
 ]) 

# Функция для предсказания, трансформации положения и записи изображения в папку transformed_images
def model_inference_final(model, path):
    to_tensor = transforms.ToTensor()

    # Load_image  -  функция  для загрузки изображений  по url  или  пути.
    image, file_name= load_image(path)
    image_trans=transform_test_valid(image)
    X=to_tensor(image_trans)
    X=X.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1).numpy()

    # ротация фотографии с учетом предсказания ((Вертикально- 0, 90'- 1, 180'- 2, 270'- 3))
    image= image.rotate(90*pred )  
    # определяем  путь для записи - папка 'transformed_images'+ имя файла( определяется функцией name_url или
    # извлекается basename(data_path) )
    save_path=os.path.join(os.getcwd(),'transformed_images', file_name)

    save_url = f'http://127.0.0.1:8000/transformed_images/{file_name}'
    image.save(save_path)

    return pred, save_path, save_url






app = FastAPI()
app.mount('/transformed_images', StaticFiles(directory="/app/transformed_images"), name='files')

# class pydentic для детекции строки
class FilePath(BaseModel):
    path: str  

@app.post('/forms/')
def model_inference(data: FilePath):
    
    pred, save_path, save_url = model_inference_final(model, data.path )
    
    return {'post_request':data.path, 'save_path':save_path, 'save_url':save_url,'pred':pred.item()}
    






if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)