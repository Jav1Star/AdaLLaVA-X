import os
# 使用镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 固定使用单核GPU（模型要求）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.adallava.eval.run_ada_llava import eval_model

model_path = "zhuoyanxu/ada-llava-L-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_name": 'ada_llava_llama',
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "latency": 1.0,
    "hardware": "nvidia_V100",
})()

eval_model(args)
