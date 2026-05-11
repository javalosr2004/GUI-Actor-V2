import base64, os
# import spaces
import json
import torch
import gradio as gr
from typing import Optional
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
from gui_actor.constants import chat_template
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference

MAX_PIXELS = 1920 * 1080

def resize_image(image, resize_to_pixels=MAX_PIXELS):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image

# @spaces.GPU
@torch.inference_mode()
def draw_point(image: Image.Image, point: list, radius=8, color=(255, 0, 0, 128)):
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    x, y = point
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        outline=color,
        width=5  # Adjust thickness as needed
    )
    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    combined = combined.convert('RGB')
    return combined

# @spaces.GPU
@torch.inference_mode()
def get_attn_map(image, attn_scores, n_width, n_height):
    """
    Build an attention heatmap for the target screenshot.

    GUI-Actor was originally written around one screenshot. If a reference image is
    also passed, attention can include visual patches from both images. The heatmap
    still needs to be reshaped to the target screenshot patch grid only, so we slice
    to the first n_width * n_height scores. This assumes the target screenshot is the
    first image in the conversation.
    """
    w, h = image.size
    expected = n_width * n_height
    scores_flat = np.array(attn_scores[0]).reshape(-1)

    print("attn len:", len(scores_flat))
    print("expected screen patches:", expected)
    print("n_width:", n_width, "n_height:", n_height)

    if len(scores_flat) < expected:
        raise ValueError(
            f"Not enough attention scores to reshape: got {len(scores_flat)}, expected {expected}"
        )

    if len(scores_flat) != expected:
        print(
            f"Attention length {len(scores_flat)} != screen patch count {expected}; "
            "using the first screen-sized slice."
        )
        scores_flat = scores_flat[:expected]

    scores = scores_flat.reshape(n_height, n_width)

    denom = scores.max() - scores.min()
    if denom == 0:
        scores_norm = np.zeros_like(scores)
    else:
        scores_norm = (scores - scores.min()) / denom

    # Resize score map to match image size
    score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST) # BILINEAR)
    # Apply colormap
    colormap = plt.get_cmap('jet')
    colored_score_map = colormap(np.array(score_map) / 255.0)  # returns RGBA
    colored_score_map = (colored_score_map[:, :, :3] * 255).astype(np.uint8)
    colored_overlay = Image.fromarray(colored_score_map)

    # Blend with original image
    blended = Image.blend(image, colored_overlay, alpha=0.3)
    return blended

# load model
if torch.cuda.is_available():
    # os.system('pip install flash-attn --no-build-isolation')
    model_name_or_path = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
    data_processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
else:
    model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
    data_processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    ).eval()

title = "GUI-Actor"
header = """
<div align="center">
    <h1 style="padding-bottom: 10px; padding-top: 10px;">🎯 <strong>GUI-Actor</strong>: Coordinate-Free Visual Grounding for GUI Agents</h1>
    <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
        Qianhui Wu*, Kanzhi Cheng*, Rui Yang*, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, Si Qin, Lars Liden<br>
        Qingwei Lin, Huan Zhang, Tong Zhang, Jianbing Zhang, Dongmei Zhang, Jianfeng Gao<br/>
    </div>
    <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
        <a href="https://microsoft.github.io/GUI-Actor/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2403.12968">📄 arXiv Paper</a> | <a href="https://github.com/microsoft/GUI-Actor">💻 Github Repo</a><br/>
    </div>
</div>
"""

theme = "soft"
css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
            #anno-img .mask.active {opacity: 0.7}"""

# @spaces.GPU
@torch.inference_mode()
def process(input_image, reference_image, instruction):
    # resize image
    if input_image is None:
        return None, "Error: please upload an input image.", None

    w_input, h_input = input_image.size
    if w_input * h_input > MAX_PIXELS:
        input_image = resize_image(input_image)

    if reference_image is not None:
        w_ref, h_ref = reference_image.size
        if w_ref * h_ref > MAX_PIXELS:
            reference_image = resize_image(reference_image)

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI agent. Given a screenshot of the current GUI (image 1) and a reference image (image 2) / human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image, # PIL.Image.Image or str to path
                    # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                }
            ],
        },
    ]

    if reference_image is not None:
        conversation[-1]["content"].append(
            {
                "type": "image",
                "image": reference_image, # PIL.Image.Image or str to path
                # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
            }
        )

    instruction_text = instruction.strip() if instruction else "Locate the matching UI element."
    conversation[-1]["content"].append({"type": "text", "text": instruction_text})

    try:
        pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
    except Exception as e:
        print(e)
        return input_image, f"Error: {e}", None
    
    px, py = pred["topk_points"][0]
    output_coord = f"({px:.4f}, {py:.4f})"

    # Use the current image size because input_image may have been resized above.
    w_draw, h_draw = input_image.size
    img_with_point = draw_point(input_image, (px * w_draw, py * h_draw))

    n_width, n_height = pred["n_width"], pred["n_height"]
    attn_scores = pred["attn_scores"]
    att_map = get_attn_map(input_image, attn_scores, n_width, n_height)
    
    return img_with_point, output_coord, att_map


with gr.Blocks(title=title, css=css) as demo:
    gr.Markdown(header)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type='pil', label='Upload input image')
            # text box
            reference_image = gr.Image(
                type='pil', label='Upload target image')
            # text box
            input_instruction = gr.Textbox(label='Instruction', placeholder='Text your (low-level) instruction here')
            submit_button = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_with_point = gr.Image(type='pil', label='Image with Point (red circle)')
            with gr.Accordion('Detailed prediction'):
                pred_xy = gr.Textbox(label='Predicted Coordinates', placeholder='(x, y)')
                att_map = gr.Image(type='pil', label='Attention Map')

    submit_button.click(
        fn=process,
        inputs=[
            input_image,
            reference_image,
            input_instruction
        ],
        outputs=[image_with_point, pred_xy, att_map]
    )

# demo.launch(debug=False, show_error=True, share=True)
# demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
demo.queue().launch(share=True)
