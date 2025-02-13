import os
import hashlib
from PIL import Image
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

#up主这里的是原生多模态模型的文件夹地址,记得填你自己的文件夹路径
model_path = "/root/autodl-tmp/deepseek-janus-pro-lora/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    language_config=language_config,
    trust_remote_code=True,
    torch_dtype=torch.float16  # 使用 float16 避免 BFloat16 问题
)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.cuda()

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def multimodal_understanding(image, question, seed=42, top_p=0.95, temperature=0.1):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.float16)  # 使用 float16
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def generate_hash_name(filename):
    """
    为文件生成哈希码名字。
    """
    hash_object = hashlib.md5(filename.encode())
    return hash_object.hexdigest()

def convert_images_to_jpeg_and_rename(folder_path):
    """
    将文件夹中的图片转换为 JPEG 格式，并重命名为哈希码名字，最后统一处理成从 1 开始的数字。
    
    :param folder_path: 图片文件夹路径
    :return: 返回新图片的文件路径列表
    """
    temp_image_paths = []
    new_image_paths = []
    count = 1  # 从 1 开始递增命名

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            
            # 打开图片并转换为 RGB 模式
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                
                # 生成哈希码名字
                hash_name = generate_hash_name(filename)
                temp_filename = f"{hash_name}.jpeg"
                temp_image_path = os.path.join(folder_path, temp_filename)
                
                # 保存为 JPEG 格式
                img.save(temp_image_path, "JPEG")
                
                # 确保新文件已保存
                if os.path.exists(temp_image_path):
                    temp_image_paths.append(temp_image_path)
                    os.remove(image_path)  # 删除原始文件
                    print(f"Converted {filename} -> {temp_filename}")
                else:
                    print(f"Failed to save {temp_filename}, skipping deletion of {filename}")
    
    # 重新命名文件为从 1 开始的数字
    for temp_image_path in sorted(temp_image_paths):
        new_filename = f"{count}.jpeg"
        new_image_path = os.path.join(folder_path, new_filename)
        os.rename(temp_image_path, new_image_path)
        new_image_paths.append(new_image_path)
        print(f"Renamed {temp_image_path} -> {new_filename}")
        count += 1

    return new_image_paths

def process_images_in_folder(folder_path, output_folder, question="Describe this picture. Please note that the person in this picture named Trump."):
    """
    处理文件夹中的所有图片，并生成对应的识别结果。
    
    :param folder_path: 图片文件夹路径
    :param output_folder: 输出结果文件夹路径
    :param question: 对每张图片提出的问题
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 将图片转换为 JPEG 格式并重命名
    new_image_paths = convert_images_to_jpeg_and_rename(folder_path)
    
    # 打印新图片路径列表
    print("New image paths:", new_image_paths)
    
    # 遍历新图片路径
    for image_path in new_image_paths:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}, skipping...")
            continue
        
        filename = os.path.basename(image_path)
        output_txt_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        
        # 加载图片
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # 调用多模态模型生成结果
        result = multimodal_understanding(image, question)
        
        # 修改结果：如果出现 "person"，在后面加上 "called Trump"
        # result = result.replace("人", "叫做川普的人")
        
        # 将结果保存到 txt 文件
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(result)
        
        print(f"Processed {filename} -> {output_txt_path}")

# 设置输入文件夹和输出文件夹
input_folder = "/root/autodl-tmp/Janus-1/trump"  # 替换为你的图片文件夹路径
output_folder = "/root/autodl-tmp/Janus-1/trump"    # 替换为输出结果文件夹路径

# 处理图片
process_images_in_folder(input_folder, output_folder, question="用中文描述这张照片,照片里的人的名字叫川普")
