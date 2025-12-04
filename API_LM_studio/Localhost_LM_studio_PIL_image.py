import time
import requests
import json
from PIL import Image
import base64
import os
import sys
from pathlib import Path
import tempfile
import io # для работы с изображениями в памяти

class LMStudioVLM:
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url
    
    def preprocess_image(self, image_input): # MODIFIED: принимает не только путь
        """Базовая предобработка изображения. Теперь принимает путь (str/Path) или объект PIL.Image."""
        img = None
        
        # Случай 1: На вход подали объект PIL.Image
        if isinstance(image_input, Image.Image):
            img = image_input
        # Случай 2: На вход подали строку пути
        elif isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            img = Image.open(image_path)
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}. Expected PIL.Image, str, or Path.")
        
        # Конвертируем изображения с прозрачностью в RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            print(f"🖼️ Конвертация из режима {img.mode} в RGB")
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def describe_multiple_images(self, image_inputs, prompt): # MODIFIED: переименовал параметр для ясности
        """Описывает несколько изображений за один запрос.
        Теперь принимает список, содержащий пути (str/Path) ИЛИ объекты PIL.Image.
        """
        start_time = time.time()
        
        try:
            # Предобработка всех изображений и кодирование в base64
            image_contents = []
            temp_files = []
            
            for i, image_input in enumerate(image_inputs): # MODIFIED: итерация по image_inputs
                print(f"🖼️ Подготовка изображения {i+1}/{len(image_inputs)}")
                
                # Предобработка изображения (теперь работает с любым типом)
                img = self.preprocess_image(image_input)
                
                # NEW: Кодируем в base64 прямо из памяти, без временного файла
                    # (Создаем "виртуальный файл" в оперативной памяти)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=100)

                #Получаем байтовое представление изображения
                img_bytes = buffered.getvalue()

                # Кодируем байты в текстовую строку base64
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                image_contents.append(img_base64)
            
            # Формируем запрос с несколькими изображениями
            content_items = []
            
            for img_base64 in image_contents:
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            content_items.append({
                "type": "text", 
                "text": prompt
            })
            
            payload = {
                "messages": [{
                    "role": "user",
                    "content": content_items
                }],
                "max_tokens": 1000,
                "temperature": 0.1,
                "stream": False
            }
            
            print(f"🤖 Отправка запроса с {len(image_inputs)} изображениями в LM Studio...")
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=600
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content'].strip()
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "output_text": description,
                    "processing_time": processing_time,
                    "image_count": len(image_inputs)
                }
            else:
                print(f"❌ Ошибка API ({response.status_code}): {response.text}")
                return {
                    "success": False,
                    "error": f"API error {response.status_code}",
                    "response_text": response.text # Добавим текст ответа для отладки
                }
                
        except Exception as e:
            print(f"❌ Ошибка при обработке изображений: {str(e)}")
            import traceback
            traceback.print_exc() # NEW: вывод полного трейса ошибки
            return {"success": False, "error": str(e)}

# для запуска файла, без main.py
if __name__ == "__main__":
    # Инициализация
    vlm = LMStudioVLM()
    
    PROJECT_ROOT = Path(__file__).parent

    # Список изображений для обработки
    image_inputs = [
        str(PROJECT_ROOT / 'Test_image' /'left_cropped_optimized_bbox.jpg'),
        str(PROJECT_ROOT / 'Test_image' /'right_cropped_optimized_bbox.jpg'),
    ]
    
    # Общий промпт для всех изображений
    prompt = f'''There are photos in front of you - screenshots with positions.
    Answer these questions about the NUMBERED elements:
    
    1) Trash can/recycle bin - which NUMBER, And on which image?
    2) Web browser - which NUMBER, And on which image? '''
    
    print("🚀 Запуск обработки всех изображений за один запрос")
    print("=" * 60)
    
    # Отправка всех 4 изображений за один раз (обращение в функцию класса)
    vlm_result=vlm.describe_multiple_images(image_inputs, prompt)

    vlm_status=vlm_result["success"]
    time=vlm_result["processing_time"]
    if vlm_result.get("success"): # cработает если success == True
        print(f"Время обработки: {time:.2f}")
        print("Позиция после обработки:", vlm_result["output_text"])
    else:
        print("Ошибка:", vlm_result.get("error", "API EROR"))
        sys.exit("Eror API LM Studio")      
