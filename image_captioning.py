
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

# Load an image
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    
def get_caption(image_path):
    image = load_image(image_path)

    # Preprocessing the Image
    img = feature_extractor(image, return_tensors="pt",tensor_type="torch").to(device)

    # Generating captions
    output = model.generate(**img)

    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption
def images_url():
    with st.form("url"):
        url = st.text_input('Enter URL of Images')
        submitted = st.form_submit_button("Submit")
        image = Image.open(requests.get(url, stream=True).raw)       
        st.image(image, caption="Image from URL.", use_column_width=True)
        if submitted:
            predicted_caption = get_caption(url)
            st.write(predicted_caption)

def main():
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("Image Caption Prediction")
    st.header('Welcome to Image Caption Prediction!')
    # st.write('This is a sample app that demonstrates the prowess of ServiceFoundry ML model deployment.🚀')
    # st.write('Visit the [Github](https://github.com/vishank97/image-captioning) repo for detailed exaplaination and to get started right away')
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

# import requests

# # Backend
# import torch

# # Image Processing
# from PIL import Image

# # Transformer and pre-trained Model
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast

# # Managing loading processing
# from tqdm import tqdm

# # Assign available GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

# # Corresponding ViT Tokenizer
# tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # Image processor
# image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# import urllib.parse as parse
# import os
# # Verify url
# def check_url(string):
#     try:
#         result = parse.urlparse(string)
#         return all([result.scheme, result.netloc, result.path])
#     except:
#         return False

# # Load an image
# def load_image(image_path):
#     if check_url(image_path):
#         return Image.open(requests.get(image_path, stream=True).raw)
#     elif os.path.exists(image_path):
#         return Image.open(image_path)
    
# def get_caption(image_path):
#     image = load_image(image_path)

#     # Preprocessing the Image
#     img = image_processor(image, return_tensors="pt",tensor_type="torch").to(device)

#     # Generating captions
#     output = model.generate(**img)

#     # decode the output
#     caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

#     return caption

# url = "https://images.pexels.com/photos/101667/pexels-photo-101667.jpeg?auto=compress&cs=tinysrgb&w=600"

# # print(get_caption(url))

# import streamlit as st
# from PIL import Image


# # Streamlit app


# def main():
#     st.title("Image Caption Generator")

#     # Upload image through file uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     # if uploaded_file is not None:
#     #     # Display the uploaded image
#     #     image = Image.open(uploaded_file)
#     #     st.image(image, caption="Uploaded Image.", use_column_width=True)

#     #     # Get the caption
#     #     caption = get_caption(image)
#     #     st.write("### Caption:")
#     #     st.write(caption)

#     # Input image URL
#     image_url = st.text_input("Or enter image URL:")
    
#     # Display the image from URL
#     image = Image.open(requests.get(image_url, stream=True).raw)
#     st.image(image, caption="Image from URL.", use_column_width=True)

#     # Get the caption
#     caption = get_caption(image_url)
#     st.write("### Caption:")
#     st.write(caption)
#         # except Exception as e:
#         #     st.warning("Error loading image from URL. Please check the URL.")

# if __name__ == "__main__":
#     main()