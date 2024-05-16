import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")

image_path = "images/house1_1.png"
image = Image.open(image_path)

if image.mode != 'RGB':
    image = image.convert('RGB')

candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = align_processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = align_model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)
