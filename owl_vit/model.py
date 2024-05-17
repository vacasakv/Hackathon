from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin
from box import Box
import matplotlib.pyplot as plt
import numpy as np


config = Box.from_yaml(filename="config/config.yaml")
amenities = config.amenities

owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
mixin = ImageFeatureExtractionMixin()

image = Image.open('images/house1_15.png')
if image.mode != 'RGB':
    image = image.convert('RGB')

inputs = owlvit_processor(text=amenities, images=image, return_tensors="pt")
outputs = owlvit_model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
results = owlvit_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = amenities
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


# Load example image
image_size = owlvit_model.config.vision_config.image_size
image_resized = mixin.resize(image, image_size)
input_image = np.asarray(image_resized).astype(np.float32) / 255.0

def plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold=0.1):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })

      
   
    
logits = torch.max(outputs["logits"][0], dim=-1)
scores = torch.sigmoid(logits.values).cpu().detach().numpy()

# Get prediction labels and boundary boxes
labels = logits.indices.cpu().detach().numpy()
boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
plot_predictions(input_image, amenities, scores, boxes, labels, score_threshold=0.1)
