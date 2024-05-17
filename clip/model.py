from PIL import Image
from transformers import CLIPModel, CLIPProcessor

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


image_path = "images/house1_1.png"
image = Image.open(image_path)


candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = clip_processor(text=candidate_labels, images=image, return_tensors="pt")

outputs = clip_model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)
