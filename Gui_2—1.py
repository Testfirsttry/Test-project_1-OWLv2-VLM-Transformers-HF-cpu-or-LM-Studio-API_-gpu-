# gui_assistant.py
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
from PIL import Image, ImageTk
import pyautogui
import json
import re
from pathlib import Path
import sys
from datetime import datetime
import time

# Добавляем пути к модулям
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))


from combined_owlv2_LM_studio_Transformers import DesktopObjectDetector
from desktop_interaction import DesktopInteraction

class DesktopAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Desktop Assistant")
        self.root.geometry("800x600")
        
        # Стиль
        self.root.configure(bg='#2b2b2b')
        
        # Переменные
        self.analysis_result = None
        self.object_positions = {}  # question_id -> object_id
        self.screen_coordinates = {}  # object_id -> (x, y)
        self.VLM_text_results = None
        self.highlight_window = None
        
        # Настройка прозрачности
        self.transparency_level = 1.0
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        self.status_bar = tk.Label(root, textvariable=self.status_var, 
                                  bg='#2b2b2b', fg='white', anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.log("Для вызова справки введите help")
        
    def create_widgets(self):
        # Основной фрейм
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Верхняя панель с кнопками
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопки действий
        self.analyze_btn = tk.Button(button_frame, text="🔍 Анализировать", 
                                    command=self.start_analysis,
                                    bg='#4CAF50', fg='white',
                                    font=('Arial', 10, 'bold'))
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.highlight_btn = tk.Button(button_frame, text="✨ Подсветить", 
                                      command=self.highlight_object,
                                      bg='#2196F3', fg='white',
                                      state=tk.DISABLED)
        self.highlight_btn.pack(side=tk.LEFT, padx=5)
        
        self.click_btn = tk.Button(button_frame, text="🖱️ Кликнуть", 
                                  command=self.click_object,
                                  bg='#FF9800', fg='white',
                                  state=tk.DISABLED)
        self.click_btn.pack(side=tk.LEFT, padx=5)
        
        self.hide_btn = tk.Button(button_frame, text="👁️ Скрыть окно", 
                                 command=self.toggle_window_visibility,
                                 bg='#9C27B0', fg='white')
        self.hide_btn.pack(side=tk.LEFT, padx=5)
        
        # Метод анализа (Transformers/LM Studio)
        method_frame = tk.Frame(button_frame, bg='#2b2b2b')
        method_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(method_frame, text="Метод:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        
        self.method_var = tk.StringVar(value="transformers")
        self.method_menu = ttk.Combobox(method_frame, textvariable=self.method_var,
                                       values=["transformers", "lm_studio"],
                                       state="readonly", width=15)
        self.method_menu.pack(side=tk.LEFT, padx=5)
        
        # Область чата/лога
        log_frame = tk.Frame(main_frame, bg='#1e1e1e')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок лога
        log_header = tk.Frame(log_frame, bg='#333', height=30)
        log_header.pack(fill=tk.X)
        tk.Label(log_header, text="📝 Лог операций", bg='#333', fg='white',
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10)
        
        # Кнопки очистки лога
        clear_btn = tk.Button(log_header, text="Очистить", command=self.clear_log,
                             bg='#555', fg='white', font=('Arial', 8))
        clear_btn.pack(side=tk.RIGHT, padx=10)
        
        # Текстовое поле лога
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                 bg='#1e1e1e', 
                                                 fg='white',
                                                 font=('Consolas', 9),
                                                 insertbackground='white')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Поле для быстрого ввода команды (опционально)
        input_frame = tk.Frame(main_frame, bg='#2b2b2b')
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(input_frame, text="Быстрая команда:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        
        self.cmd_var = tk.StringVar()
        cmd_entry = tk.Entry(input_frame, textvariable=self.cmd_var, 
                            bg='#555', fg='white', width=50)
        cmd_entry.pack(side=tk.LEFT, padx=5)
        
        cmd_btn = tk.Button(input_frame, text="Выполнить", 
                           command=self.execute_command,
                           bg='#607D8B', fg='white')
        cmd_btn.pack(side=tk.LEFT)
        
        # Привязываем Enter к выполнению команды
        cmd_entry.bind('<Return>', lambda e: self.execute_command())
    
    def log(self, message, level="INFO"):
        """Добавляет сообщение в лог с цветом"""
        colors = {
            "INFO": "white",
            "SUCCESS": "#4CAF50",  # зеленый
            "WARNING": "#FF9800",  # оранжевый
            "ERROR": "#F44336",    # красный
            "DEBUG": "#9C27B0"     # фиолетовый
        }
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        self.log_text.insert(tk.END, formatted_msg + "\n", level)
        self.log_text.tag_config(level, foreground=colors.get(level, "white"))
        self.log_text.see(tk.END)
        
        # Обновляем статус бар для важных сообщений
        if level in ["SUCCESS", "ERROR"]:
            self.status_var.set(message)
    
    def clear_log(self):
        """Очищает лог"""
        self.log_text.delete(1.0, tk.END)
        self.log("Лог очищен", "INFO")
    
    def start_analysis(self, object_list=None):
        """Запускает анализ в отдельном потоке"""
        object_list=object_list

        # 1 Отключаем кнопки на время анализа
        self.analyze_btn.config(state=tk.DISABLED)
        self.highlight_btn.config(state=tk.DISABLED)
        self.click_btn.config(state=tk.DISABLED)
        
        self.log("🚀 Запуск анализа...", "INFO")
        if object_list == None:
            self.log(f"""Стандартные объекты:
"Trash can/recycle bin", "Web browser"
                 
Для выбора иных объектов:
Введите Find "Trash can", "Web browser", "Notepad" """)
        
        # 2. Запуск в ДВУХ (2) отдельных потоках 1)Анализа и 2)временной "Прозрачности"

        # снижение прозрачности "-alpha" в отдельном потоке, 
        # для срабатывания взятие скриншота в run_analysis_thread 
        screenshot_thread = threading.Thread(target=self.make_transparent_for_screenshot)
        screenshot_thread.daemon = True
        screenshot_thread.start()

        # Запускаем в отдельном потоке чтобы не блокировать GUI
        thread = threading.Thread(target=self.run_analysis_thread, args=(object_list,))
        thread.daemon = True
        thread.start()

    def make_transparent_for_screenshot(self):
        """Делает окно прозрачным на короткое время для скриншота"""

        self.root.after(0, lambda: self.root.wm_attributes("-alpha", 0.1))
        time.sleep(1.5)  
        self.root.after(0, lambda: self.root.wm_attributes("-alpha", 1.0))
        time.sleep(1.1)

    def run_analysis_thread(self, object_list=None):
        """Запускает анализ в отдельном потоке управления"""
        try:
            object_list= object_list
            if object_list!= None:
                self.log(f"\n Выбранные объекты: {object_list}", "INFO")
            # 1. получение информации о выборе метода "transformers / LM_srtudio"
            method = self.method_var.get()
            self.log(f"\n🔧 Выбран метод: {method}", "INFO")
            
            
            # 2. Создаем экземпляр детектора
            detector = DesktopObjectDetector()
            
            # 3. Запускаем полный пайплайн из файла combined_owlv2_LM_studio_Transformers.py
            result = detector.run_full_pipeline(analysis_method=method,
                                                show_math_plot_fig = "show",
                                                show_final_results="show",
                                                input_items=object_list
                                                )
            """
#что выходит из run_full_pipeline

run_full_pipeline = final_result = {
               "screenshot_path": screenshot_path,
               "split_images": [left_path, right_path],
               "owl_results": owl_results,
               "image_parts": image_parts,        

               "vlm_result_all": vlm_result_all, # все данные после analyze_with (7 пункт)
               
                        #что выходит из vlm_result_all
                        vlm_result_all= return{
                                        "method": "lm_studio",
                                        "output_text": vlm_result_all["output_text"],
                                        "processing_time": vlm_result_all["processing_time"],
                                        "raw_result": vlm_result_all}

               # не работает, похоже на кривое обращение"VLM_output_text": vlm_result_all['output_text'], #текстовое описание VLM
               # тоже не"VLM_processing_time": vlm_result_all['processing_time'], # время обработки VLM

               "object_positions": object_id, # словарь c id и номером запроса {1: 12, 2: 7}
               "analysis_method": analysis_method, #transformers или lm_studio
               "input_items": input_items #входной набор объектов
               }
            """
            
            self.log("\n✅ Анализ завершен!", "SUCCESS")

            # Сохраняем результаты
            self.analysis_result = result
            
            # Получение ID обьектов из ответа
            time_vlm=result["vlm_result_all"]['processing_time']
            self.log(f"\n⏱️ Время обработки VLM: {time_vlm}")

            total_detections = sum(count['detection_count'] for count in result["owl_results"])
            self.log(f"\n📊 Всего обнаружено объектов: {total_detections}")

            self.object_positions = result["object_positions"] # словарь c id и номером запроса {1: 12, 2: 7}
            self.log(f"\n📍 Найдены объекты: {self.object_positions}") 

            self.VLM_text_results = result["vlm_result_all"]['output_text']
            print("VLM_text_results", self.VLM_text_results)
            self.log(f"\n📝 Ответ VLM: {self.VLM_text_results}")

            # Включаем кнопки после анализа
            self.analyze_btn.config(state=tk.NORMAL)
            self.highlight_btn.config(state=tk.NORMAL)
            self.click_btn.config(state=tk.NORMAL)
            
            # Показываем найденные объекты
            if self.object_positions:
                self.log(" Найденные объекты:", "INFO")
                for q_id, obj_id in self.object_positions.items():
                    obj_name = "Корзина" if q_id == 1 else "Браузер"
                    self.log(f"  {obj_name}: ID {obj_id}", "SUCCESS")
            
        except Exception as e:
            self.log(f"❌ Ошибка анализа: {str(e)}", "ERROR")

            #Включает только кнопку анализа
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Восстанавливаем окно в случае ошибки
            self.root.wm_attributes("-alpha", 1.0)

    
    def highlight_object(self,object_id):
        """Подсвечивает найденный объект ID -> json -> координаты"""

        # 1. получение id из полученных ранее в def start_analysis()
        object_id = object_id #словарь c id и номером запроса {1: 12, 2: 7}
        
        # 2. Создаем экземпляр класса для интеракции
        highlight = DesktopInteraction()

        # 3. по номерам передача ID, и параметра - 1)подсветки и 2)клика по области
        for object in object_id():
            highlight.process_object(object_id = object,
                                    highlight="show",
                                    сlick_on_object="hide")

        if not self.object_id:
            self.log("Нет координат для подсветки", "WARNING")
            return
        
        """
        # Закрываем предыдущее окно подсветки
        if self.highlight_window:
            try:
                self.highlight_window.destroy()
            except:
                pass
        
        # Создаем окно подсветки для первого объекта
        obj_id = list(self.object_positions.values())[0]
        coords = self.screen_coordinates.get(obj_id)
        
        if coords:
            self.create_highlight_window(coords, obj_id)
        """
        """
    def create_highlight_window(self, coords, obj_id):
        #Создает окно для подсветки объекта (через matplotlib)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            x, y = coords
            
            # Создаем фигуру
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_alpha(0.7)  # Прозрачный фон
            
            # Рисуем круг
            circle = patches.Circle((0.5, 0.5), 0.4, 
                                   facecolor='red', 
                                   alpha=0.5,
                                   edgecolor='yellow',
                                   linewidth=3)
            ax.add_patch(circle)
            
            # Добавляем текст
            ax.text(0.5, 0.5, f'ID: {obj_id}', 
                   ha='center', va='center', 
                   fontsize=14, fontweight='bold',
                   color='white')
            
            # Настраиваем вид
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Позиционируем окно рядом с объектом
            plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y-100)}")
            plt.get_current_fig_manager().window.attributes('-topmost', True)
            
            # Сохраняем ссылку на окно
            self.highlight_window = plt
            
            self.log(f"✨ Подсветка объекта {obj_id} на ({x:.0f}, {y:.0f})", "SUCCESS")
            
            # Автоматически закрываем через 3 секунды
            self.root.after(3000, lambda: plt.close() if plt else None)
            
            plt.show(block=False)
            
        except ImportError:
            # Альтернатива без matplotlib - простое окно tkinter
            self.create_simple_highlight(coords, obj_id)
    
    def create_simple_highlight(self, coords, obj_id):
        #Простая подсветка через tkinter
        x, y = coords
        
        # Создаем окно поверх всех окон
        highlight = tk.Toplevel(self.root)
        highlight.overrideredirect(True)
        highlight.attributes('-topmost', True)
        highlight.attributes('-alpha', 0.7)
        
        # Красный круг
        highlight.configure(bg='red')
        
        # Размер и позиция
        size = 100
        highlight.geometry(f"{size}x{size}+{int(x-size/2)}+{int(y-size/2)}")
        
        # Метка с ID
        label = tk.Label(highlight, text=f"ID: {obj_id}", 
                        bg='red', fg='white',
                        font=('Arial', 12, 'bold'))
        label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.highlight_window = highlight
        
        # Автозакрытие
        self.root.after(3000, highlight.destroy)
"""
    def click_object(self):
        """Кликает по найденному объекту ID -> json -> координаты"""

        # 1. получение id из полученных ранее в def start_analysis()
        object_id = object_id #словарь c id и номером запроса {1: 12, 2: 7}
        
        # 2. Создаем экземпляр класса для интеракции
        highlight = DesktopInteraction()

        # 3. по номерам передача ID, и параметра - 1)подсветки и 2)клика по области
        for object in object_id():
            highlight.process_object(object_id = object,
                                    highlight="show", #оставлю подсветка для наглядности места клика
                                    сlick_on_object="show")

        if not self.object_id:
            self.log("Нет координат для подсветки", "WARNING")
            return

    def toggle_window_visibility(self):
        #Переключает видимость окна, за счет сравнения значения transparency_level
        if self.transparency_level == 1.0:
            # Делаем окно прозрачным
            self.transparency_level = 0.3
            self.root.wm_attributes("-alpha", self.transparency_level)
            self.hide_btn.config(text="👁️ Показать окно")
            self.log("Окно стало полупрозрачным", "INFO")
        else:
            # Восстанавливаем видимость
            self.transparency_level = 1.0
            self.root.wm_attributes("-alpha", self.transparency_level)
            self.hide_btn.config(text="👁️ Скрыть окно")
            self.log("Окно восстановлено", "INFO")
    
    def execute_command(self):
        """Выполняет быструю команду"""
        cmd = self.cmd_var.get().strip()
        if not cmd:
            return
        
        self.log(f"> {cmd}", "DEBUG")
        
        # Простые команды
        if cmd.startswith("click"):
            # Пример: "click 25" - клик по объекту 25
            try:
                obj_id = int(cmd.split()[1])
                # Здесь можно добавить логику клика по конкретному ID
                self.log(f"Команда: клик по объекту {obj_id}", "INFO")
            except:
                self.log("Использование: click [ID]", "WARNING")
        
        elif cmd == "clear":
            self.clear_log()

        elif cmd == "hide":
            self.toggle_window_visibility()
        

        elif cmd == "start_analysis":
            self.start_analysis()

        elif cmd.startswith("Find"):
        # Варианты: # start_analysis
        # Find "Trash can", "Web browser", "Notepad"
            params = cmd[len("Find"):].strip()

            # Регулярка для извлечения объектов (учитывает кавычки и запятые)
            pattern = r'\"([^\"]+)\"|([^,\s]+)' #работает на "Trash can", "Web browser", "Notepad"
            matches = re.findall(pattern, params)
                
            # Объединяем результаты
            objects = []
            for match in matches:
                # match[0] - если в кавычках, match[1] - если без кавычек
                obj = match[0] if match[0] else match[1]
                if obj:
                    objects.append(obj)
                
            if objects:
                self.start_analysis(object_list=objects)
            else:
                self.log("Не указаны объекты для анализа", "WARNING")
   
        elif cmd == "help":
            self.log(f"""Команды:          
clear - очистить окно log             
hide - переключить видимость окна 
                                   
start_analysis - начать анализ
start_analysis Trash can, Web browser            
или              
start_analysis "Trash can", "Web browser", "Notepad" - для поиска определенных объектов
                                  
click [ID] - где id номер обнаруженного объекта""")
        
        else:
            self.log(f"Неизвестная команда: {cmd}", "WARNING")
        
        # Очищаем поле ввода
        self.cmd_var.set("")

def main():
    root = tk.Tk()
    
    app = DesktopAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()