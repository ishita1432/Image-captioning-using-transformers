
import streamlit as st 
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch 
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import os
import requests
import urllib.parse as parse


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images",accept_multiple_files=True,type=["png","jpg","jpeg","jfif"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images,False)
            for i,caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)

def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    
def get_caption(image_path):
    image = load_image(image_path)
    img = feature_extractor(image, return_tensors="pt",tensor_type="torch").to(device)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption
    
def images_url():
    with st.form("url"):
        url = st.text_input('Enter URL of Images')
        submitted = st.form_submit_button("Submit")
        if url and check_url(url):
            image = Image.open(requests.get(url, stream=True).raw)
            st.image(image, caption="Image from URL.", use_column_width=True)
            if submitted:
                predicted_caption = get_caption(url)
                st.write(predicted_caption)
        elif submitted:
            st.write("Please enter a valid URL.")


def main():
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("Image Caption Prediction")
    st.header('Welcome to Image Caption Prediction! 🚀')
    tab1, tab2= st.tabs(["Image from computer", "Image from URL"])
    with tab1:
        image_uploader()
    with tab2:
        images_url()

def predict_step(images_list):
    images = []
    for image in tqdm(images_list):
        i_image = Image.open(image)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
     
    return preds

if __name__ == '__main__':
    main()

