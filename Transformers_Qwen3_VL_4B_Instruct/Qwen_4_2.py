from transformers import AutoModelForImageTextToText, AutoProcessor
import time
import torch
from pathlib import Path
from PIL import ImageGrab

torch.set_num_threads(12) # i5 12400f

def main_qwen3(model_path, image_path, text_input):
    model_path = model_path

    start_time = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        dtype="auto",    
        #dtype=torch.float16, #потребление памяти чуть больше чем в dtype="auto" (в пике, но средне такое же)
                            #`torch_dtype` is deprecated!
                            #  auto = bfloat16 плавно загружает память
        device_map="cpu",   
        #low_cpu_mem_usage=True,    #вроде бы нет разницы
        #attn_implementation="sdpa",  #для 844x589 и 1920*1080 разницы нет.
    )

    load_time = time.time() - start_time
    print(f"✅ Модель загружена за {load_time:.2f} сек")

    processor = AutoProcessor.from_pretrained(model_path)
    
    content_items = []

    for image in image_path:
        content_items.append({
        "type": "image",
        "image": image  # или base64_image, если нужно кодировать
        })

    content_items.append({
    "type": "text", 
    "text": text_input
    })

    messages = [
    {
        "role": "user",
        "content": content_items
    }
    ]

    """
    messages = [
        {
            "role": "user",
            "content": [
                    { "type": "image","image": image_path},
                {"type": "text", "text": text_input},
            ],
        }
    ]
    """

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    print("🧠 Generation...")

    inference_start = time.time()

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, #num_beams=1, #словно немного медленнее на 1%?
        max_new_tokens=300,    #do_sample=False # вылезает ошибка The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    inference_time = time.time() - inference_start

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    #словарь для вывода
    return {"output_qwen3_text":output_text,
            "generation_time":inference_time           
            }

# для запуска файла, без main.py
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).parent
    TEST_MODEL_PATH_QWEN = PROJECT_ROOT

    screenshot = ImageGrab.grab()

    TEST_IMAGE_PATH = [
        str(PROJECT_ROOT / 'Test_image' /'left_cropped_optimized_bbox.jpg'),
        str(PROJECT_ROOT / 'Test_image' /'right_cropped_optimized_bbox.jpg'),
    ]
    main_qwen3(
    model_path=TEST_MODEL_PATH_QWEN, 
    image_path=TEST_IMAGE_PATH, 

    text_input=f'''There are photo in front of you - screenshot with positions.
     Answer these questions about the NUMBERED elements:

     1. Trash can/recycle bin - which NUMBER, And on which image??
     2. Web browser - which NUMBER, And on which image??

     ANSWER FORMAT:

     Answer preparation: [text with analysis about the positions of objects]

     Final result:
     1: [number]
     2: [number]''' 
    )
    qwen3_text=qwen3_resilt["output_qwen3_text"]
    qwen3_Tine_to_run=qwen3_resilt["generation_time"]
    print(f"Время обработки: {qwen3_Tine_to_run:.2f}")
    print("Позиция после обработки:",qwen3_text)


