from PIL import ImageGrab, Image
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent
print("PATH =  ",PROJECT_ROOT)
#PROJECT_ROOT= r'E:\LLM_models_20_11_25\new1'

#screenshot = ImageGrab.grab(bbox=(0, 0, 500, 500)) # Сделать скриншот всего экрана
screenshot = ImageGrab.grab()
#IMAGE_PATHS = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]

"""
Image.copy() #копирует
Image.crop() #обрезает 
"""
# Сохранить скриншот в файл
IMAGE_SAVE_DIR = PROJECT_ROOT / 'image_save'
IMAGE_SAVE_DIR.mkdir(exist_ok=True)  # Создаем директорию если не существует
IMAGE_PATH = IMAGE_SAVE_DIR / 'new1.png'
screenshot.save(IMAGE_PATH) 

#Создадим 2 копии изображения, что работы с ними
image_1, image_2= screenshot.copy(), screenshot.copy()

# сделаем из 1920x1080 2 изображения 1080x1080 левое и 1080x1080 правое  - для 3-х целей: 
     # 1. квадратное изображение для OWLv2 (при прямоугольном смещение bbox - bounding box)
     # 2. для создания перекрытия, при разделении 
          # (середина 2-х изображений будет на 960, но с доп пикселями 1080 - 960 = 12.5% наложения)
     # 3. проще разбить квадратное изображение, чем прямоугольное (возможно надуманно)

#crop area (left, top, right, bottom) (как x1,y1,x2,y2)
left_cropped = image_1.crop((0, 0, 1080, 1080))
right_cropped = image_2.crop((1920-1080, 0, 1920, 1080))

# сохраним пути новых изображений
left_image_path, right_image_path = IMAGE_SAVE_DIR / 'left_cropped.png', IMAGE_SAVE_DIR / 'right_cropped.png'
left_cropped.save(left_image_path)
right_cropped.save(right_image_path)

text_queries = [
    ["desktop icon", "application icon", "shortcut icon"],
    ["window", "application window", "browser window"],
    #["taskbar", "start menu", "system tray"],
    #["button", "close button", "minimize button", "maximize button"],
    #["menu bar", "title bar", "status bar", "scroll bar"],
    #["file explorer", "folder icon", "document icon"],
    #["notification area", "search bar", "address bar"]
    ["blue icon", "green icon", "red icon", "yellow icon"],                     #доп для обнаружения
    ["small square icon", "large rectangular window", "thin horizontal bar"],   #доп
    #["everything visible", "all UI elements", "all clickable items"],           #доп
    #["text label", "title bar text", "menu text"]                               #доп
]

OWLv2_MODEL_PATH = PROJECT_ROOT / 'owlv2_large_patch14_ensemble'
OUTPUT_DIR_OWLV2 = PROJECT_ROOT / 'Output_OWLv2'


#вызов главной функции, для работы OWLv2
from owlv2_large_patch14_ensemble.owlv2_5_5 import main_owl

#массив с результатами после работы owlv2 для каждого фото
owl_result=[]
image_set=[left_image_path, right_image_path]

print("Левое и правое изображения\n",image_set,"\n")

for i in range(len(image_set)):
     owl_result.append(
          main_owl(
     model_path= OWLv2_MODEL_PATH,
     image_path=image_set[i],
     text_queries=text_queries,
     output_path=OUTPUT_DIR_OWLV2
          )
     )
# результаты обработки OWLV2 - 2 словаря [ {} , {} ]
#owl_visualization_path = owl_result['visualization_path']   # расположение Image файла с bbox
                                                                      #(optimized_bbox.jpg)
#owl_json_path= owl_result['json_path']                      # расположение Json файла
#owl_txt_path = owl_result['txt_path']                      # расположение Txt файла
#owl_detection_count = owl_result['detection_count']        № количество объектов детекции


all_parts = []
for result in owl_result:
     img = Image.open(result['visualization_path'])
        
        # Разбиваем на 4 части без сохранения
     overlap = 54
     parts = [
          img.crop((0, 0, 540 + overlap, 540 + overlap)),        # левый верх с перекрытием
          img.crop((540 - overlap, 0, 1080, 540 + overlap)),     # правый верх с перекрытием
          img.crop((0, 540 - overlap, 540 + overlap, 1080)),     # левый низ с перекрытием  
          img.crop((540 - overlap, 540 - overlap, 1080, 1080))   # правый низ с перекрытием
        ]
     all_parts.extend(parts)
print(all_parts)    


"""Mathplot, for 8 image"""
all_parts_with_names = [(f"part_{i+1}", img) for i, img in enumerate(all_parts)]

def show_all_parts_with_names(image_parts_with_names, title="Все части с именами"):
    """Показывает все части с именами"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(image_parts_with_names):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{name}\n{img.size}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Использование:
show_all_parts_with_names(all_parts_with_names, "Части с именами")



text_search_objects=f'''There are photo in front of you - screenshot with positions.
     Answer these questions about the NUMBERED elements:

     1. Trash can/recycle bin - which NUMBER?
     2. Web browser - which NUMBER?

     ANSWER FORMAT:
     1: [number]
     2: [number]

     Use 0 if not found. Only numbers from the image.'''

"""API_LM_studio"""
from API_LM_studio.Localhost_LM_studio_PIL_image import LMStudioVLM
vlm = LMStudioVLM()

vlm_result = vlm.describe_multiple_images(
     image_inputs=all_parts,
     prompt=text_search_objects,
)

if vlm_result.get("success"): # Неявная передача значения. cработает если success == True
     print("Позиция после обработки:", vlm_result["output_text"])
else:
     print("Ошибка:", vlm_result.get("error", "API EROR"))
     sys.exit("Eror API LM Studio")   

vlm_Tine_to_run=vlm_result["processing_time"]
