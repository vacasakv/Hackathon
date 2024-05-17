import streamlit as st
from PIL import Image
from box import Box
import numpy as np
from utils.helpers import create_prompts, align_detect_amenity, clip_detect_amenity
from align.model import align_processor, align_model
from clip.model import clip_processor, clip_model
from owl_vit.model import owlvit_processor, owlvit_model, mixin, plot_predictions
import torch
st.set_option('deprecation.showPyplotGlobalUse', False)

config = Box.from_yaml(filename="config/config.yaml")
amenities = config.amenities
models = {'clip': (clip_processor, clip_model),
          'align': (align_processor, align_model)}

st.write("# Amenity Detection")
amenities = st.multiselect("Amenities to detect: ", amenities)

uploaded_file = st.file_uploader(":red[Choose an image...]")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if st.button("Detect"):
        pos_prompts, neg_prompts = create_prompts(amenities)

        col1, col2 = st.columns(2)

        for i, model_name in enumerate(models.keys()):
            processor, model = models[model_name]
            with col1 if i % 2 == 0 else col2:
                st.header(model_name.upper())   

                subcol_1, mid, subcol_2 = st.columns([1, 0.01, 7.5])
                with subcol_1:
                    for amenity in amenities:
                        st.image(f"icons/{amenity.replace(' ', '_')}.png")
                
                for pos, neg in zip(pos_prompts, neg_prompts):
                    if model_name == 'clip':
                        predicted_prompt, probs = clip_detect_amenity(
                             image=image, pos_prompt=pos, neg_prompt=neg, processor=processor, model=model
                         )
                    elif model_name == 'align':
                        predicted_prompt, probs = align_detect_amenity(
                            image=image, pos_prompt=pos, neg_prompt=neg, processor=processor, model=model
                        )
       
                    with subcol_2:
                        if predicted_prompt == pos:
                            st.write(f":green[{predicted_prompt.replace('there is ', '')} yes]")
                        else:
                            st.write(f":red[{predicted_prompt.replace('there is ', '')}]")

        # Owl-ViT
        st.header("Owl-ViT")
        inputs = owlvit_processor(text=amenities, images=image, return_tensors="pt")
        outputs = owlvit_model(**inputs)

        image_size = owlvit_model.config.vision_config.image_size
        image_resized = mixin.resize(image, image_size)
        input_image = np.asarray(image_resized).astype(np.float32) / 255.0

        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
        fig = plot_predictions(input_image, amenities, scores, boxes, labels)

        st.pyplot(fig)