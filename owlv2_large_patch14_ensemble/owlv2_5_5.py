import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection
#import random
import os
#import numpy as np
import json
from datetime import datetime
torch.set_num_threads(12) # i5 12400f  12 tread CPU
#import matplotlib.pyplot as plt

# ====== УЛУЧШЕННЫЕ НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ======
VISUALIZATION_SETTINGS = {
    "number_position": "top_left",  # Варианты: "top_left", "top_right", "bottom_left", "bottom_right", "center"
    "number_size": 16,               # Размер шрифта для номеров
    "bbox_padding": "fixed",      # "fixed" или "adaptive" - адаптивные отступы
    "fixed_padding": 6,              # Фиксированный отступ в пикселях
    "show_debug_info": False,       # Показывать дополнительную отладочную информацию
    "bbox width": 3                 # толщина линии
}

FILTER_SETTINGS = {
    "max_area_ratio": 0.05,     # 5% от площади изображения
    "min_area_ratio": 0.0005,    # 0.05% от площади изображения (в 1080p папка чуть больше)
    "min_width_pixels": 15,     # Минимальная ширина 15 пикселей
    "min_height_pixels": 15,    # Минимальная высота 15 пикселей
    "max_width_ratio": 0.33,     # Максимальная ширина 33% от изображения
    "max_height_ratio": 0.33,    # Максимальная высота 33% от изображения
    "iou_threshold": 0.3,       # Порог для объединения пересекающихся bbox (Intersection over Union (IoU))
                                # чем ближе к 1, тем нужно больше пересечения
    "min_confidence": 0.10,      # минимальная уверенность
    "max_aspect_ratio": 4.0,  # Максимальное соотношение ширины к высоте
    "min_aspect_ratio": 0.25   # Минимальное соотношение ширины к высоте
}

"""
def get_number_position(box, text_width, text_height, position_type):
    #Вычисляет оптимальную позицию для номера внутри bbox
    x1, y1, x2, y2 = box
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    padding_x = -VISUALIZATION_SETTINGS['fixed_padding']
    padding_y = -VISUALIZATION_SETTINGS['fixed_padding']

    positions = {
        "top_left": (x1 + padding_x, y1 + padding_y),
        "top_right": (x2 - text_width - padding_x, y1 + padding_y),
        "bottom_left": (x1 + padding_x, y2 - text_height - padding_y),
        "bottom_right": (x2 - text_width - padding_x, y2 - text_height - padding_y),
        "center": (
            x1 + (bbox_width - text_width) / 2,
            y1 + (bbox_height - text_height) / 2
        )
    }
    
    return positions.get(position_type, positions["top_left"])
"""

def get_rich_color(i):
    """Генерирует цвет, гарантируя что хотя бы один канал < 100"""
    # Генерируем базовые значения как раньше
    r = (i * 67) % 200 + 55
    g = (i * 131) % 200 + 55
    b = (i * 241) % 200 + 55
    
    # 🔧 ГАРАНТИРУЕМ, что хотя бы один канал < 100
    # Если все три канала >= 100, выбираем случайный и уменьшаем его
    if r >= 100 and g >= 100 and b >= 100:
        # Выбираем какой канал уменьшить (детерминированно на основе i)
        channel_to_reduce = i % 3
        if channel_to_reduce == 0:
            r = (i * 67) % 45 + 55  # 55-99
        elif channel_to_reduce == 1:
            g = (i * 131) % 45 + 55  # 55-99
        else:
            b = (i * 241) % 45 + 55  # 55-99
    
    return (r, g, b)

def create_optimized_visualization(image, all_detections_list, output_path,start_id=1):
    """Создает оптимизированную визуализацию с улучшенным позиционированием номеров"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    font_regular = ImageFont.truetype("arial.ttf", VISUALIZATION_SETTINGS["number_size"])
    
    # Предварительно вычисляем размеры всех текстов
    text_sizes = {}
    current_id = start_id # передача id - для предотвращения дублирования, при >1 вызове owlv2()
    for i, det in enumerate(all_detections_list):
        text = str(current_id)
        text_bbox = draw.textbbox((0, 0), text, font=font_regular)
        text_sizes[current_id] = {
            'width': text_bbox[2] - text_bbox[0],
            'height': text_bbox[3] - text_bbox[1]
        }
        current_id += 1
    
    # Рисуем каждый bbox с оптимально расположенным номером
    current_id = start_id
    for i, det in enumerate(all_detections_list):
        box = det['box']
        x1, y1, x2, y2 = box
        
        # Генерируем цвет
        color = get_rich_color(current_id)
        
        # Вычисляем отступ и создаем расширенный bbox для отображения
        padding = VISUALIZATION_SETTINGS['fixed_padding']
        display_box = [
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding)
        ]
        
        # Рисуем расширенный bbox
        draw.rectangle(display_box, outline=color, width=VISUALIZATION_SETTINGS["bbox width"])
        
        # Получаем позицию для номера
        text_size = text_sizes[current_id]
        """
        text_x, text_y = get_number_position(
            box, text_size['width'], text_size['height'],
            VISUALIZATION_SETTINGS["number_position"]
        )
        """
        text_x = x1 - padding
        text_y = y1 - padding

        # Корректируем позицию, чтобы не выходить за границы изображения
        text_x = max(0, min(text_x, image.width - text_size['width']))
        text_y = max(0, min(text_y, image.height - text_size['height']))
        background_bbox = [
            text_x , text_y+1 ,      # ← ЛЕВЫЙ ВЕРХНИЙ УГОЛ
            text_x + text_size['width'], text_y + text_size['height'] +6 # ПРАВЫЙ НИЖНИЙ УГОЛ
        ]

        # Рисуем скругленный прямоугольник для фона
        #draw.rounded_rectangle(background_bbox, fill=color)
        draw.rounded_rectangle(background_bbox, fill="black")

        # Рисуем номер
        draw.text((text_x, text_y), str(current_id), fill='white', font=font_regular)
        
        # Отладочная информация (если включена)
        if VISUALIZATION_SETTINGS["show_debug_info"]:
            debug_text = f"{det['label']} ({det['score']:.2f})"
            debug_y = y2 + 5
            if debug_y + 15 < image.height:
                draw.text((x1, debug_y), debug_text, fill=color, font=font_regular)
        
        current_id += 1
    
    # Сохраняем изображение
    draw_image.save(output_path)
    print(f"  🎯 Оптимизированная визуализация сохранена: {output_path}")
    
    return draw_image

# ====== ОСТАЛЬНЫЕ ФУНКЦИИ (filter_boxes_by_multiple_criteria, calculate_iou, merge_overlapping_boxes, export_detections_to_json, export_detections_to_text) остаются без изменений ======
def filter_boxes_by_multiple_criteria(detections, image_size):
    """Фильтрует bbox по нескольким критериям размера"""
    image_width, image_height = image_size
    image_area = image_width * image_height
    
    max_area = image_area * FILTER_SETTINGS["max_area_ratio"]
    min_area = image_area * FILTER_SETTINGS["min_area_ratio"]
    max_width = image_width * FILTER_SETTINGS["max_width_ratio"]
    max_height = image_height * FILTER_SETTINGS["max_height_ratio"]
    min_width = FILTER_SETTINGS["min_width_pixels"]
    min_height = FILTER_SETTINGS["min_height_pixels"]
    
    filtered_detections = []
    removal_stats = {
        "too_large": 0,
        "too_small": 0,
        "wrong_shape": 0
    }
    
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Проверяем все критерии
        if area > max_area:
            removal_stats["too_large"] += 1
            continue
            
        if area < min_area:
            removal_stats["too_small"] += 1
            continue
            
        if width < min_width or height < min_height:
            removal_stats["too_small"] += 1
            continue
            
        if width > max_width or height > max_height:
            removal_stats["too_large"] += 1
            continue
        
        # Дополнительная проверка на очень вытянутые формы
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > FILTER_SETTINGS["max_aspect_ratio"] or aspect_ratio < FILTER_SETTINGS["min_aspect_ratio"]:
            removal_stats["wrong_shape"] += 1
            continue
            
        filtered_detections.append(det)
    
    # Вывод статистики фильтрации
    if any(removal_stats.values()):
        print(f"  🗑️ Фильтрация: слишком большие {removal_stats['too_large']}, "
              f"маленькие {removal_stats['too_small']}, "
              f"неправильная форма {removal_stats['wrong_shape']}")
    
    return filtered_detections

def calculate_iou(box1, box2):
    """Вычисляет Intersection over Union (IoU) двух bbox"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def merge_overlapping_boxes(all_detections, iou_threshold=FILTER_SETTINGS["iou_threshold"]):
    """Объединяет пересекающиеся bbox с улучшенным алгоритмом"""
    if not all_detections:
        return all_detections
    
    # Сортируем по уверенности (от высокой к низкой)
    sorted_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)
    merged_detections = []
    merged_count = 0
    
    while sorted_detections:
        best_det = sorted_detections.pop(0)
        merged_detections.append(best_det)
        
        non_overlapping = []
        for det in sorted_detections:
            iou = calculate_iou(best_det['box'], det['box'])
            if iou < iou_threshold:
                non_overlapping.append(det)
            else:
                merged_count += 1
        
        sorted_detections = non_overlapping
    
    if merged_count > 0:
        print(f"  🔗 Объединено {merged_count} пересекающихся bbox")
    
    return merged_detections

#start_id=1 -значение по умолчанию, но заменяется передаваемым в функцию?
def export_detections_to_json(all_detections_list, output_path, start_id=1):
    """Экспортирует информацию о bbox в JSON"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(all_detections_list),
        "detections": []
    }
    current_id = start_id
    for i, det in enumerate(all_detections_list, 1):
        detection_info = {
            "id": current_id,   # используем глобальный счетчик вместо локального
            "label": det['label'],
            "score": det['score'],
            "coordinates": {
                "x1": det['box'][0],
                "y1": det['box'][1],
                "x2": det['box'][2],
                "y2": det['box'][3],
                "width": det['box'][2] - det['box'][0],
                "height": det['box'][3] - det['box'][1]
            },
            "center_point": {
                "x": (det['box'][0] + det['box'][2]) / 2,
                "y": (det['box'][1] + det['box'][3]) / 2
            }
        }
        export_data["detections"].append(detection_info)
        current_id += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data

def export_detections_to_text(all_detections_list, output_path, start_id=1):
    """Экспортирует информацию в читаемый текстовый формат"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ПРОНУМЕРОВАННЫЕ BBOX ОБЪЕКТЫ\n")
        f.write("=" * 50 + "\n\n")
        
        current_id = start_id
        for i, det in enumerate(all_detections_list, 1):
            f.write(f"ОБЪЕКТ #{current_id}:\n")
            f.write(f"  Метка: {det['label']}\n")
            f.write(f"  Уверенность: {det['score']:.3f}\n")
            f.write(f"  Координаты: [{det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f}]\n")
            f.write(f"  Ширина: {det['box'][2] - det['box'][0]:.1f}px\n")
            f.write(f"  Высота: {det['box'][3] - det['box'][1]:.1f}px\n")
            f.write(f"  Центр: ({((det['box'][0] + det['box'][2]) / 2):.1f}, {((det['box'][1] + det['box'][3]) / 2):.1f})\n")
            f.write("-" * 30 + "\n")
            current_id += 1

# ====== ОСНОВНОЙ КОД С ВЫБОРОМ СТИЛЯ ВИЗУАЛИЗАЦИИ ======
def main_owl(model_path, image_path, text_queries, output_path, start_id=1):
    # Загрузка модели и обработка изображения (код из предыдущего примера)
    model_path = model_path
    
    processor = Owlv2Processor.from_pretrained(model_path, use_fast=True)
    model = Owlv2ForObjectDetection.from_pretrained(model_path)

    image_path = image_path

    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    print(f"📏 Размер изображения: {image.size}")


    text_queries_list= text_queries
# Обработка всех запросов
    all_detections = []

    for i, query_set in enumerate(text_queries_list):
        print(f"\n🔍 Запрос {i+1}/{len(text_queries_list)}: {query_set}")
        
        try:
            inputs = processor(text=[query_set], images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs, interpolate_pos_encoding=True)

            target_sizes = torch.tensor([[image.height, image.width]])
            
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=FILTER_SETTINGS["min_confidence"]
                )

            # Сбор детекций
            result = results[0]
            current_count = 0
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                if score > FILTER_SETTINGS["min_confidence"]:
                    box_coords = [round(i, 2) for i in box.tolist()]
                    label_text = query_set[label] if label < len(query_set) else f"unknown_{label}"
                    
                    detection_info = {
                        'box': box_coords,
                        'score': round(score.item(), 3),
                        'label': label_text,
                        'query_source': query_set[0]
                    }
                    all_detections.append(detection_info)
                    current_count += 1
            
            print(f"  📊 Найдено объектов: {current_count}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    print(f"\n📈 Всего исходных обнаружений: {len(all_detections)}")

    # УЛУЧШЕННАЯ ФИЛЬТРАЦИЯ
    filtered_detections = filter_boxes_by_multiple_criteria(all_detections, image.size)
    print(f"📏 После улучшенной фильтрации: {len(filtered_detections)}")

    final_detections = merge_overlapping_boxes(filtered_detections)
    print(f"🤝 После объединения пересекающихся: {len(final_detections)}")
    # После получения final_detections:
    
    print(f"\n{'='*60}")
    print("🎨 НАСТРОЙКИ ВИЗУАЛИЗАЦИИ")
    print(f"{'='*60}")
    
    # Можно вручную выбрать позицию или использовать настройки по умолчанию
    selected_position = VISUALIZATION_SETTINGS["number_position"]
    print(f"Выбрана позиция: {selected_position}")
    
    #output_dir = os.path.dirname(image_path)
    output_dir = output_path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Основная визуализация с выбранной позицией
    main_image_path = os.path.join(output_dir, f"{base_name}_optimized_bbox.jpg")
    create_optimized_visualization(original_image, final_detections, main_image_path, start_id)

    # 2. Экспорт данных
    json_path = os.path.join(output_dir, f"{base_name}_bbox_data.json")
    export_detections_to_json(final_detections, json_path, start_id)
    
    txt_path = os.path.join(output_dir, f"{base_name}_bbox_data.txt") 
    export_detections_to_text(final_detections, txt_path, start_id)
    
    print(f"\n📁 Результаты сохранены:")
    print(f"   🎯 Основная визуализация: {main_image_path}")
    print(f"   📊 JSON с координатами: {json_path}")
    print(f"   📝 Текстовые данные: {txt_path}")

    #возвращаем словарь значений
    return {'visualization_path': main_image_path,
            'json_path': json_path,
            'txt_path': txt_path,
            'detection_count': len(final_detections),
            'image_size': image.size,
            'next_start_id': start_id + len(final_detections)
    }

if __name__ == "__main__":
    from pathlib import Path
    from PIL import ImageGrab

        # Тестовые параметры для запуска напрямую
    PROJECT_ROOT = Path(__file__).parent
    TEST_MODEL_PATH = PROJECT_ROOT #/ 'owlv2_large_patch14_ensemble'

        #image создание скриншота
    screenshot = ImageGrab.grab()
    TEST_IMAGE_PATH = PROJECT_ROOT / 'Test_image' / 'Test_image1.png'
    
    screenshot.save(TEST_IMAGE_PATH) # Сохранить скриншот в файл

    # список запросок, по которым модель OWLv2 будет обрабатывать изобоажение
        # новая строка запроса - новая итерация обработки
            # итерации суммируются и проходят очистку
    TEST_QUERIES = [
    ["desktop icon", "application icon", "shortcut icon"],
    ["window", "application window", "browser window"],
    #["taskbar", "start menu", "system tray"],
    #["button", "close button", "minimize button", "maximize button"],
    #["menu bar", "title bar", "status bar", "scroll bar"],
    #["file explorer", "folder icon", "document icon"],
    #["notification area", "search bar", "address bar"]
    ["blue icon", "green icon", "red icon", "yellow icon"],                     #доп для обнаружения
    #["small square icon", "large rectangular window", "thin horizontal bar"],   #доп
    #["everything visible", "all UI elements", "all clickable items"],           #доп
    #["text label", "title bar text", "menu text"]                               #доп
    ]
    
    TEST_OUTPUT_PATH= PROJECT_ROOT / 'Test_output'
    #TEST_IMAGE_PATH = r'E:\LLM_models_20_11_25\new1\owlv2_large_patch14_ensemble\Test_image\Test_image1 — 960x580.png' #960x580(1)
    #TEST_IMAGE_PATH = r'E:\LLM_models_20_11_25\new1\owlv2_large_patch14_ensemble\Test_image\n5.png' #960x580(2)
    # Вызываем функцию с тестовыми данными
    test_results =  main_owl(
        model_path=TEST_MODEL_PATH,
        image_path=TEST_IMAGE_PATH,
        text_queries=TEST_QUERIES,
        output_path = TEST_OUTPUT_PATH,
        start_id=1
    )
