import json
import tkinter as tk
import pyautogui
import time
from pathlib import Path
from typing import Dict, Optional

class DesktopInteraction:
    """
    Упрощенная версия DesktopInteraction
    tkinter для подсветки
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.output_dir = self.project_root / 'Output_OWLv2'
        
        # для коррекции смещения, при разббивки изображения
        self.right_offset = 840  # 1920 - 1080
    
    def _correct_coordinates(self, obj: Dict) -> Dict:
        """коррекция координаты для правого изображения
        при создании json координаты были как 0,0 без учета смещения."""
        corrected = obj.copy()
        
        # Корректируем все x-координаты
        if 'coordinates' in corrected:
            coords = corrected['coordinates']
            for key in ['x1', 'x2']:
                if key in coords:
                    coords[key] += self.right_offset
        
        # Корректируем center_point
        if 'center_point' in corrected:
            center = corrected['center_point']
            if 'x' in center:
                center['x'] += self.right_offset
        
        return corrected
    
    def load_json_data(self) -> Dict[str, list]:
        """Загрузка данные из двух JSON файлов"""
        data = {'left': [], 'right': []}
        
        # Левый файл
        left_file = self.output_dir / 'left_cropped_bbox_data.json'
        if left_file.exists():
            with open(left_file, 'r') as f:
                data['left'] = json.load(f).get('detections', [])
        
        # Правый файл
        right_file = self.output_dir / 'right_cropped_bbox_data.json'
        if right_file.exists():
            with open(right_file, 'r') as f:
                data['right'] = json.load(f).get('detections', [])
        
        return data
    
    def find_object(self, object_id: int) -> Optional[Dict]:
        """поиск объекта по ID в загруженных данных"""
        data = self.load_json_data()
        
        # Определяем, в каком файле искать
        left_ids = [obj['id'] for obj in data['left']]
        #присваиваем наибольшее значение id
        max_left_id = max(left_ids) if left_ids else 0
        
        if object_id <= max_left_id:
            # Ищет в левом файле(left_cropped_bbox_data.json), 
            # если id меньше чем максимальный, в левом фото (json)
            for obj in data['left']:
                if obj['id'] == object_id: # если id совпадает с обнаруженным ViT
                    #print(f"Найден объект ID {object_id} в левом файле")
                    return obj
        else:
            # Ищет в "правом" файле и корректируем координаты (было коррекции json)
            for obj in data['right']:
                if obj['id'] == object_id:
                    #print(f"Найден объект ID {object_id} в правом файле")
                    return self._correct_coordinates(obj)
        
        print(f"❌ Объект ID {object_id} не найден")
        return None
    
    
    def highlight_with_tkinter(self, x1: float, y1: float, 
                               x2: float, y2: float, 
                               duration: int = 2):
        """
        Упрощенная подсветка через tkinter
        Создает маленькое окно только над нужной областью
        """
        try:
            # создание окна
            window = tk.Tk()
            window.overrideredirect(True)  # Без рамок
            window.attributes('-topmost', True)  # Поверх всех окон
            window.attributes('-alpha', 0.3)  # Полупрозрачность
            
            # Размеры области
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            # Позиционируем окно над областью
            window.geometry(f"{width}x{height}+{int(x1)}+{int(y1)}")
            
            # Заливаем цветом
            canvas = tk.Canvas(window, bg='yellow', highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            # Автоматическое закрытие
            window.after(duration * 2000, window.destroy)
            
            # Запускаем окно
            window.mainloop()
            
            print(f"✨ Подсвечена область: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
            
        except Exception as e:
            print(f"⚠️ Ошибка подсветки: {e}")
    
    def click_center(self, obj: Dict):
        """Кликает в центр объекта"""
        # получение координаты центра
        
        x = obj['center_point']['x']
        y = obj['center_point']['y']

        # Проверяет границы экрана
        screen_width, screen_height = pyautogui.size()
        if 0 <= x <= screen_width and 0 <= y <= screen_height:
            # Перемещаем и кликаем
            pyautogui.moveTo(x, y, duration=0.5)
            time.sleep(0.2)
            pyautogui.click()
            print(f"✅ Клик по координатам: ({x:.0f}, {y:.0f})")
        else:
            print(f"❌ Координаты вне экрана: ({x:.0f}, {y:.0f})")
    


     #-----------------#
    # Основная функция #
    #-----------------#
    def process_object(self, object_id: int, highlight="show", сlick_on_object="show"):
        """
        Основной метод: находит, подсвечивает и кликает по объекту
        """
        print(f"\n🎯 Обработка объекта ID {object_id}")
        
        # 1. Находим объект
        obj = self.find_object(object_id)
        if not obj:
            return
        
        # 2. Получаем координаты bbox
        coords = obj.get('coordinates', {})
        if not coords:
            print("❌ У объекта нет координат")
            return
        
        # 3. Подсвечиваем область
        if highlight == "show":
            
            self.highlight_with_tkinter(
                coords['x1'], coords['y1'],
                coords['x2'], coords['y2'],
                duration=2
            )
        
        # 4. Пауза перед кликом
        time.sleep(0.5)
        
        # 5. Кликаем по центру
        if сlick_on_object == "show":
            self.click_center(obj)
    
    def run_demo(self):
        """Демонстрация работы, c вводом объектов"""
        print("Запуск демонстрации")
        
        # Запрашиваем ID объектов
        print("\n Введите ID объектов через запятую (например: 21,7):")
        user_input = input("> ").strip()
        
        if not user_input:
            print("❌ Не указаны объекты")
            return
        
        # Обрабатываем каждый объект
        for obj_id_str in user_input.split(','):
            try:
                obj_id = int(obj_id_str.strip())
                print("Значение id=",obj_id)

                self.process_object(obj_id,)
                # Пауза между объектами
                time.sleep(1)
                
            except ValueError:
                print(f"⚠️ Некорректный ID: {obj_id_str}")
            except Exception as e:
                print(f"❌ Ошибка обработки объекта: {e}")

# Пример использования
if __name__ == "__main__":
    interactor = DesktopInteraction()
    interactor.run_demo()