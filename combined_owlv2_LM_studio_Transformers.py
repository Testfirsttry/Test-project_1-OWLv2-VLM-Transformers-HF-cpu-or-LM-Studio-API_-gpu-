from PIL import ImageGrab, Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import json
from typing import List, Dict, Tuple, Optional
import re

class DesktopObjectDetector:
     def __init__(self, project_root: Optional[Path] = None):
          """Инициализация"""
          self.PROJECT_ROOT = project_root or Path(__file__).parent
          #print(f"PATH = {self.PROJECT_ROOT}")
        
          # Создаем директории
          self.IMAGE_SAVE_DIR = self.PROJECT_ROOT / 'image_save'
          self.OUTPUT_DIR_OWLV2 = self.PROJECT_ROOT / 'Output_OWLv2'
          self.IMAGE_SAVE_DIR.mkdir(exist_ok=True)
          self.OUTPUT_DIR_OWLV2.mkdir(exist_ok=True)
        
        # Настройки
          self.text_queries = [
          ["desktop icon", "application icon", "shortcut icon"],
          ["window", "application window", "browser window"],
          #["taskbar", "start menu", "system tray"],
          #["button", "close button", "minimize button", "maximize button"],
          #["menu bar", "title bar", "status bar", "scroll bar"],
          #["file explorer", "folder icon", "document icon"],
          #["notification area", "search bar", "address bar"]
          ["blue icon", "green icon", "red icon", "yellow icon"],                     #доп для обнаружения
          #["small square icon", "large rectangular window", "thin horizontal bar"],   #доп
          ["everything visible", "all UI elements", "all clickable items"],           #доп
          #["text label", "title bar text", "menu text"]                               #доп
        ]
        
          # Пути к моделям
          self.OWLv2_MODEL_PATH = self.PROJECT_ROOT / 'owlv2_large_patch14_ensemble'
          self.Qwen3_VL_MODEL_PATH = self.PROJECT_ROOT / 'Transformers_Qwen3_VL_4B_Instruct'
    
     def capture_screenshot(self) -> Path:
          screenshot = ImageGrab.grab()
          image_path = self.IMAGE_SAVE_DIR / 'new1.png'
          screenshot.save(image_path)
          print(f"✅ Скриншот сохранен: {image_path}")
          return image_path
    
     def split_into_two_squares(self, image_path: Path) -> Tuple[Path, Path]:
          """Разбивка изображения 1920x1080 на два квадрата 1080x1080"""
          image = Image.open(image_path)
          image_1, image_2 = image.copy(), image.copy()
          
          left_cropped = image_1.crop((0, 0, 1080, 1080))
          right_cropped = image_2.crop((1920-1080, 0, 1920, 1080))
          
          left_image_path = self.IMAGE_SAVE_DIR / 'left_cropped.png'
          right_image_path = self.IMAGE_SAVE_DIR / 'right_cropped.png'
          
          left_cropped.save(left_image_path)
          right_cropped.save(right_image_path)
          
          #print("Левое изображение и Правое:\n" {left_image_path},"\n" {right_image_path}")
          return left_image_path, right_image_path
    
     def process_with_owlv2(self, image_paths: List[Path], start_id = 1) -> List[Dict]:
          """Обрабатка изображения через OWLv2"""
          from owlv2_large_patch14_ensemble.owlv2_5_5 import main_owl
          
          results = []
          current_id = start_id
          
          for image_path in image_paths:            
               result = main_owl(
                    model_path=self.OWLv2_MODEL_PATH,
                    image_path=image_path,
                    text_queries=self.text_queries,
                    output_path=self.OUTPUT_DIR_OWLV2,
                    start_id=current_id,
               )
               results.append(result)

               current_id += result['detection_count']
               #print(f"Найдены объекты: {result['detection_count']}, следующий ID: {current_id}")
          
          return results
    
     def split_owl_results_into_parts(self, owl_results: List[Dict]) -> List[Image.Image]:
          """Разбивка результаты OWLv2 на 8 частей с перекрытием"""        
          all_parts = []
          overlap = 54
          
          for result in owl_results:
               img = Image.open(result['visualization_path'])
               # Разбиваем на 4 части без сохранения
               parts = [
                    img.crop((0, 0, 540 + overlap, 540 + overlap)),        # левый верх
                    img.crop((540 - overlap, 0, 1080, 540 + overlap)),     # правый верх
                    img.crop((0, 540 - overlap, 540 + overlap, 1080)),     # левый низ
                    img.crop((540 - overlap, 540 - overlap, 1080, 1080))   # правый низ
               ]
               all_parts.extend(parts)

          #print(f"Всего получено частей: {len(all_parts)}")
          return all_parts
    
     def show_all_parts_with_names(self, image_parts: List[Image.Image], title: str = "Все части с именами"):
          """Вывод 8 частей изображения в matplotlib"""
          
          image_parts_with_names = [(f"part_{i+1}", img) for i, img in enumerate(image_parts)]
          
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
    
     def analyze_with_transformers(self, image_parts: List[Image.Image], query_text: str) -> Dict:
          """Анализ через Qwen 3 VL Transformers (CPU)"""
          from Transformers_Qwen3_VL_4B_Instruct.Qwen_4_2 import main_qwen3
          
          qwen3_result = main_qwen3(
               model_path=self.Qwen3_VL_MODEL_PATH,
               image_path=image_parts,
               text_input=query_text,
          )

          return {
               "method": "transformers",
               "output_text": qwen3_result["output_qwen3_text"],
               "processing_time": qwen3_result["generation_time"],
               "raw_result": qwen3_result
          }
    
     def analyze_with_lm_studio(self, image_parts: List[Image.Image], query_text: str) -> Dict:
          """Анализ через Qwen 3 VL LM Studio API (GPU)"""          
          from API_LM_studio.Localhost_LM_studio_PIL_image import LMStudioVLM
          vlm = LMStudioVLM()
               
          vlm_result_all = vlm.describe_multiple_images(
                    image_inputs=image_parts,
                    prompt=query_text,
          )
          
          # cработает если success == True
          if vlm_result_all.get("success"):
               return {
                    "method": "lm_studio",
                    "output_text": vlm_result_all["output_text"],
                    "processing_time": vlm_result_all["processing_time"],
                    "raw_result": vlm_result_all
               }
          else: 
               return {
                    "error": vlm_result_all.get("error", "Ошибка API")
               }

    
     def extract_object_positions(self, analysis_text: str) -> Dict[int, int]:
          """Извлекает id bbox объектов из текста анализа"""
          #возвращает по одному (1) ID на позицию (формат 1: 12, 2: 7)
          positions = {}

          # Убирает лишние пробелы и разделяем на строки
          lines = analysis_text.strip().split('\n')
          
          for line in lines:
               line = line.strip()
               # Пропуск пустых строк
               if not line:
                    continue
               # поиск строки, которые выглядят как "X: Y" где X - номер, Y - число или список
               if ':' in line:
                    try:
                         # Разделяем на номер вопроса и значение
                         question_part, value_part = line.split(':', 1)
                         
                         # Извлечение номер вопроса (берем только цифры)
                         question_match = re.search(r'\d+', question_part)
                         if not question_match:
                              continue
                              
                         question_num = int(question_match.group())
                         
                         # Извлечение ID объекта - несколько вариантов формата
                         # Форматы: "1: 123", "1: [123]", "1: [123, 456]", "ID 1: 123"
                         
                         # Убираем квадратные скобки если есть
                         value_part = value_part.replace('[', '').replace(']', '').strip()
                         
                         # Ищем первое число в значении
                         value_match = re.search(r'\d+', value_part)
                         if value_match:
                              object_id = int(value_match.group())
                              #прямое присвоение элемента к словарю
                              positions[question_num] = object_id
                              print(f"✅ Вопрос {question_num}: объект ID {object_id}")
                              #print("positions результаты",positions)
                         else:
                              print(f"⚠️ Не удалось извлечь ID из строки: {line}")
                              
                    except (ValueError, IndexError, AttributeError) as e:
                         print(f"❌ Ошибка обработки строки '{line}': {e}")
                         continue
          
          return positions
     
     def print_final_results(self, results: Dict):
        """Выводит итоговые результаты"""
        print("\n" + "=" * 60)
        print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
          
        # Информация об анализе
        analysis = results["vlm_result_all"]
        print(f"🔧 Метод анализа: {results['analysis_method']}")
        print(f"⏱️ Время обработки: {analysis['processing_time']:.2f} сек")
          
        print(f"\n📝 Ответ VLM:\n{analysis['output_text']}")
               
        # Позиции объектов
        positions = results["object_positions"]
        input_items = results["input_items"]
        if positions:
            print("\n📍 Найденные позиции:")
            for question_num in sorted(positions.keys()):
                object_id = positions[question_num]
                obj_name = f"Объект {question_num}"
                if question_num - 1 < len(input_items):
                     obj_name = input_items[question_num - 1]
                
                print(f"  {obj_name} (вопрос {question_num}): ID {object_id}")
        else:
            print("\n❌ Не удалось определить позиции объектов")
          
        # Информация о детекциях
        total_detections = sum(result['detection_count'] for result in results["owl_results"])
        print(f"\n📊 Всего обнаружено объектов: {total_detections}")
    
    
     def get_detection_by_id(self, owl_results: List[Dict], object_id: int) -> Optional[Dict]:
          """Находит детекцию по ID во всех результатах OWLv2"""
          for result in owl_results:
               with open(result['json_path'], 'r') as f:
                    data = json.load(f)
               
               for detection in data['detections']:
                    if detection['id'] == object_id:
                         return detection
          return None
    
     def get_coordinates_for_click(self, detection: Dict) -> Tuple[float, float]:
          """Получает координаты для клика из детекции"""
          coords = detection['coordinates']
          center_x = (coords['x1'] + coords['x2']) / 2
          center_y = (coords['y1'] + coords['y2']) / 2
          return center_x, center_y
    

      #------------------------#
     ### Основной цикл работы ##
      #-----------------------#
     def run_full_pipeline(self, analysis_method: str = "transformers", 
                           show_math_plot_fig = "show",
                           show_final_results="show",
                           input_items=None) -> Dict:
          if input_items is None:
               input_items = ["Trash can/recycle bin", "Web browser"]
     
          """Запускает полный пайплайн обработки"""
          print("=" * 60)
          print("Start")
          print("=" * 60)
          
          # 1. Захват скриншота
          screenshot_path = self.capture_screenshot()
          
          # 2. Разбивка на два квадрата
          left_path, right_path = self.split_into_two_squares(screenshot_path)
          
          # 3. OWLv2 детекция
          owl_results = self.process_with_owlv2([left_path, right_path])
          
          # 4. Разбивка результатов на части
          image_parts = self.split_owl_results_into_parts(owl_results)
          
          # 5. Показываем части (опционально)
          if show_math_plot_fig == "show":
               self.show_all_parts_with_names(image_parts, "Части для анализа Qwen 3 VL")
          
          # 6. Текстовый запрос для анализа
          # Формируем шаблон ответа
          questions = "\n".join([
          f"{i+1}. {item.replace('_', ' ')} - which NUMBER?" 
          for i, item in enumerate(input_items)
          ])

          result_template = "\n".join([
          f"{i+1}: [number]" 
          for i in range(len(input_items))
          ])

          query_text = f'''There are photos in front of you - screenshots with numbered elements.
            Answer these questions about the numbered objects:

            {questions}

            ANSWER FORMAT:
            Answer preparation: [Concise analysis of object positions and numbers]
            Final result:
            {result_template}'''

          print(query_text)
        
          # 7. выбор метода анализа и передача частей изображения + запроса текстом
          if analysis_method == "transformers":
               vlm_result_all = self.analyze_with_transformers(image_parts, query_text)
               """return{
                    "method": "transformers",
                    "output_text": qwen3_result["output_qwen3_text"],
                    "processing_time": qwen3_result["generation_time"],
                    "raw_result": qwen3_result}"""
               
          elif analysis_method == "lm_studio":
               vlm_result_all = self.analyze_with_lm_studio(image_parts, query_text)
               """return{
                    "method": "lm_studio",
                    "output_text": vlm_result_all["output_text"],
                    "processing_time": vlm_result_all["processing_time"],
                    "raw_result": vlm_result_all}"""
          
          else:
               raise ValueError("Неверный метод анализа. Выберите 'transformers' или 'lm_studio'")
          
          # 8. Извлечение id bbox объектов
          object_id = {}
          object_id = self.extract_object_positions(vlm_result_all["output_text"])
          
          # 9. Формируем итоговый результат
          final_result = {
               "screenshot_path": screenshot_path,
               "split_images": [left_path, right_path],
               "owl_results": owl_results,
               "image_parts": image_parts,        

               "vlm_result_all": vlm_result_all, # все данные после analyze_with (7 пункт)
               "object_positions": object_id, # словарь c id и номером запроса {1: 12, 2: 7}
               "analysis_method": analysis_method, #transformers или lm_studio
               "input_items": input_items, #входной набор объектов
               "VLM_output_text": vlm_result_all['output_text'], #текстовое описание VLM
               "VLM_processing_time": vlm_result_all['processing_time'], # время обработки VLM
                              # что выходит из vlm_result_all
                              #     vlm_result_all = return{
                              #          "method": "lm_studio",
                              #          "output_text": vlm_result_all["output_text"],
                              #          "processing_time": vlm_result_all["processing_time"],
                              #          "raw_result": vlm_result_all}
          }

          # 10. Вывод результатов
          if show_final_results == "show":
               self.print_final_results(final_result)
          
          return final_result


# для запуска файла, без main.py
if __name__ == "__main__":
      # 1. Создаем экземпляр класса
     detector = DesktopObjectDetector()

     #detector.run_full_pipeline("transformers")
     detector.run_full_pipeline("lm_studio",
                                show_math_plot_fig = "hide",
                                show_final_results = "show")
