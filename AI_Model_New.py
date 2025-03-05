# Import necessary libraries
from transformers import CLIPProcessor, CLIPModel # type: ignore
import torch # type: ignore
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Set up the local directory containing images
image_folder = r'C:\Users\Smith\Desktop\CapstoneProject\BackgroundImages'  # Update to your local path

# List of image filenames in the local directory
# image_filenames = [
#     'Boots.jpg',
#     'Dress_shoes.jpg',
#     'High_Heels.jpg',
#     'Loafer.jpg',
#     'Runners.jpg',
#     'Sandals.jpg',
#     'Stilettos.jpg'
# ]

# Dynamically list all image files in the folder
image_filenames = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Generate full paths for the images
image_paths = [os.path.join(image_folder, filename) for filename in image_filenames]

print(f"Found {len(image_paths)} images in {image_folder}.")

# Load CLIP pre-trained model and processor
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# Function to create an image grid
def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols  # Calculate number of rows
    w, h = imgs[0].size  # Get width and height of the first image
    grid = Image.new('RGB', size=(cols * w, rows * h))  # Create a blank image grid
    for i, img in enumerate(imgs):
        x = (i % cols) * w  # Calculate column position
        y = (i // cols) * h  # Calculate row position
        grid.paste(img, box=(x, y))  # Paste image at the calculated position
    return grid

# Append the images into a list
images = []
for filename in image_filenames:
    img_path = os.path.join(image_folder, filename)  # Construct full path
    try:
        images.append(Image.open(img_path))  # Open and append image
    except FileNotFoundError:
        print(f"Error: {img_path} not found. Please check the file path and name.")

# Display the images as a grid
if images:
    grid = image_grid(images, cols=3)
    grid.show()  # Display the image grid

# List of class names for different types of shoes
classes = [
    'Dress Shoes', 'Loafer', 'Boots (Ankle)', 'Boots (Knee High)',
    'High Heels', 'Stilettos', 'Sandals', 'Runners of shoes'
]

# Process the text (class names) and images into tensors suitable for the model
# inputs = processor(
#     text=classes,  # List of class names
#     images=images,  # List of images
#     return_tensors="pt",  # Return PyTorch tensors
#     padding=True,  # Pad the inputs to the same length
#     do_convert_rgb=False  # Assume images are already in RGB format
# )
######new block
inputs = processor(
    text=classes,  # List of class names
    images=images,  # List of images
    return_tensors="pt",  # Return PyTorch tensors
    padding=True  # Pad the inputs to the same length
)
######new block


# Pass the processed inputs through the model to get the outputs
outputs = model(**inputs)

# Extract the logits for image-text similarity
logits_per_image = outputs.logits_per_image

# Apply softmax to the logits to get probabilities for each class
probs = logits_per_image.softmax(dim=1)

# Display probabilities as bar charts
fig = plt.figure(figsize=(8, 20))  # Create a figure
for idx in range(len(images)):
    # Display the original image
    ax1 = fig.add_subplot(len(images), 2, 2 * idx + 1)
    ax1.imshow(images[idx])  # Show the image
    ax1.axis('off')  # Remove axis for the image

    # Display the probabilities as a horizontal bar chart
    ax2 = fig.add_subplot(len(images), 2, 2 * idx + 2)
    ax2.barh(range(len(probs[0])), probs[idx].detach().numpy(), tick_label=classes)
    ax2.set_xlim(0, 1.0)  # Set x-axis limits to range from 0 to 1

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)

# Display the figure with all subplots
plt.show()
# Local directory containing the images
image_folder = r'C:\Users\Smith\Desktop\CapstoneProject\DamageImages'  # Updated local path

# List of image filenames in the local directory
# image_filenames = [

#     'Broken stitches.jpg',  # Filename of the image showing broken stitches
#     'dirty shoes.jpg',  # Filename of the image showing a dirty item
#     'Leather faded.jpg',  # Filename of the image showing faded leather
#     'Opened seam.jpg',  # Filename of the image showing an opened seam
#     'Slanted heel.jpg',  # Filename of the image showing a slanted heel
#     'Sole not flat.jpg',  # Filename of the image showing a sole that is not flat
#     'tears.jpg',  # Filename of the image showing tears
#     'Weak cementing.jpg',  # Filename of the image showing weak cementing
#     'Weak cementing1.jpg',  # Another filename of the image showing weak cementing
#     'wrinkles.jpg',  # Filename of the image showing wrinkles
# ]



# Dynamically list all image files in the folder
damage_filenames = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Generate full paths for the images
damage_image_paths = [os.path.join(image_folder, filename) for filename in damage_filenames]

print(f"Found {len(damage_image_paths)} images in {image_folder}.")

# Load CLIP pre-trained model and processor
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# Initialize an empty list to store the images
images = []
for filename in damage_image_paths:
#for filename in image_filenames:
    img_path = os.path.join(image_folder, filename)  # Construct the full path to the image file
    try:
        images.append(Image.open(img_path))  # Open and append the image
    except FileNotFoundError:
        print(f"Error: {img_path} not found. Please check the file path and name.")

# Function to create an image grid
def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols  # Calculate the number of rows
    w, h = imgs[0].size  # Get the size of the first image
    grid = Image.new('RGB', size=(cols * w, rows * h))  # Create a blank grid
    for i, img in enumerate(imgs):
        x = (i % cols) * w  # Calculate the column position
        y = (i // cols) * h  # Calculate the row position
        grid.paste(img, box=(x, y))  # Paste the image at the calculated position
    return grid

# Create and display the image grid
if images:
    grid = image_grid(images, cols=3)
    grid.show()  # Display the image grid

# Define the list of classes (labels) corresponding to different types of damage
classes = [
    'Broken stitches',
    'Opened seam',
    'Slanted heel',
    'Sole not flat',
    'Peeling leather/tears',
    'Weak cementing',
    'wrinkles',
    'dirty shoes',
    'Leather Faded',
]

# Process the text (class labels) and images to create the inputs for the model
inputs = processor(
    text=classes,  # List of class labels
    images=images,  # List of images to be classified
    return_tensors="pt",  # Return the data as PyTorch tensors
    padding=True,  # Pad the inputs to the same length
    do_convert_rgb=False,  # Assume images are already in RGB format
)

# Pass the inputs through the model to get the outputs
outputs = model(**inputs)

# Extract the logits for image-text similarity scores
logits_per_image = outputs.logits_per_image  # This is the image-text similarity score

# Apply softmax to the logits to get the probabilities for each class
probs = logits_per_image.softmax(dim=1)  # Take the softmax to get the label probabilities

# Get the highest scoring label for each image
predicted_labels = torch.argmax(probs, dim=1)

# Plotting the results
fig = plt.figure(figsize=(8, 20))
for idx in range(len(images)):
    # Display the original image
    ax1 = fig.add_subplot(len(images), 2, 2 * idx + 1)
    ax1.imshow(images[idx])
    ax1.axis('off')  # Remove axis for a cleaner look

    # Display the probabilities as a horizontal bar chart
    ax2 = fig.add_subplot(len(images), 2, 2 * idx + 2)
    ax2.barh(range(len(probs[0].detach().numpy())), probs[idx].detach().numpy(), tick_label=classes)
    ax2.set_xlim(0, 1.0)  # Set x-axis limits to range from 0 to 1

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)

# Display the final plot
plt.show()

# Adjust subplot parameters for spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)
# Display the figure with all subplots
plt.show()

"""### Point to craftperson
1. Heel Repair -> Category with Heel & Slanted Heels
2. Half Sole -> Weak cementing
3. Full sole -> Weak cementing
4. Patch & Sewing Repair -> Broken stitches & Opened seam
5. Hardware Repair -> Sole not flat, wrinkles
6. Insole Repair
7. Internet Heel or Linear Repair
8. Leather Care (Cleaning, Polishing) -> Dirty shoes
9. Leather Customization (RE-DYE) -> Leather faded
10. Leather Alternation (Adjustment of uppers, etc., widen calf of the top) -> Peeling leather/tears
"""

# Print the results
for i, label_idx in enumerate(predicted_labels):
    # Print the filename of the image and the predicted class label
    print(f"Image {damage_image_paths[i]} is predicted to be: {classes[label_idx]}")
    # Provide specific recommendations based on predicted classes
    if str(classes[label_idx]) == 'Broken stitches':
        print("Should be: Patch & Sewing Repair")
    elif str(classes[label_idx]) == 'Opened seam':
        print("Should be: Patch & Sewing Repair")
    elif str(classes[label_idx]) == 'Slanted heel':
        print("Should be: Heel Repair")
    elif str(classes[label_idx]) == 'Sole not flat':
        print("Should be: Hardware Repair")
    elif str(classes[label_idx]) == 'Peeling leather/tears':
        print("Should be: Leather Alternation")
    elif str(classes[label_idx]) == 'Weak cementing':
        print("Should be: Half/Full sole")
    elif str(classes[label_idx]) == 'wrinkles':
        print("Should be: Hardware Repair")
    elif str(classes[label_idx]) == 'dirty shoes':
        print("Should be: Leather Care")
    elif str(classes[label_idx]) == 'Leather Faded':
        print("Should be: Leather Customization RE-DYE")
    else:
        print("There are no specific recommendations for this class")

"""# Demo
Demo 1: Correct to match
- Structure:
    1. User input
    2. Image import
    3. Classify damage of shoes
    4. Classify type of shoes
    5. Classify material of shoes
    6. Comparing input and model output section

Demo 2: Fail to match
- Structure:
    1. User input
    2. Image import
    3. Classify damage of shoes
    4. Classify type of shoes
    5. Classify material of shoes
    6. Comparing input and model output section
"""

# Customer input
input_customer_type = "Dress Shoes"
input_customer_restoration = "Leather Customization RE-DYE"

# String variables for storing outcome
damage = ""
material = ""
shoes_type = ""

"""Predefined function to display information for backend"""

def print_info(control, predicted_type, predicted_damage, predicted_material, 
               input_customer_type, input_customer_restoration):
    if control == "correct":
        print("The information is correct:")
        print("Shoes type:", predicted_type)
        print("Shoes damage:", predicted_damage)
        print("Shoes material:", predicted_material)
    elif control == "wrong":
        print("The information:")
        print("The customer input:")
        print("Type:", input_customer_type)
        print("Damage:", input_customer_restoration)
        print("The model output:")
        print("Shoes type:", predicted_type)
        print("Shoes damage:", predicted_damage)
        print("Shoes material:", predicted_material)

"""## Image import"""

# Local directory containing the images
image_folder = r'C:\Users\Smith\Desktop\CapstoneProject\DamageImages'

# List of image filenames in the local directory
damage_image_paths = [
    'Leather faded.jpg',  # Filename of the image showing faded leather
]

images = []
for filename in damage_image_paths:
    img_path = os.path.join(image_folder, filename)  # Construct the full path to the image file
    images.append(Image.open(img_path))  # Open the image file and append it to the images list

# Assuming image_grid is defined elsewhere in the script or imported
grid = image_grid(images, cols=3)  # Create an image grid with 3 columns
grid.show()  # Display the image grid using .show()

"""## Classify damage"""
# List of damage classes (labels)
classes = [
    'Broken stitches',
    'Opened seam',
    'Slanted heel',
    'Sole not flat',
    'Peeling leather/tears',
    'Weak cementing',
    'wrinkles',
    'Dirty shoes',
    'Leather Faded'
]

# Process text (classes) and images using a transformer pipeline (processor)
inputs = processor(
    text=classes,  # List of damage classes
    images=images,  # List of images to process
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Pad inputs to the same length
    do_convert_rgb=False  # Assume images are already in RGB format
)

# Pass inputs through the model to get outputs
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)

# Extract logits for image-text similarity scores
logits_per_image = outputs.logits_per_image  # Tensor of similarity scores

# Apply softmax to logits to get class probabilities
probs = logits_per_image.softmax(dim=1)  # Probabilities for each class label

# Get the highest scoring label (class) for each image
predicted_damage = torch.argmax(probs, dim=1)  # Tensor of predicted class indices

# Convert predicted class index to damage label string
damage = classes[predicted_damage.item()]  # Get the damage label for the predicted class

# Print the predicted damage label
print(f"Predicted Damage: {damage}")

"""## Classify type"""
# List of shoe types (labels)
classes = [
    'Dress Shoes',
    'Loafer',
    'Boots (Ankle)',
    'Boots (Knee High)',
    'High Heels',
    'Stilettos',
    'Sandals',
    'Runners of shoes'
]

# Process text (classes) and images using a transformer pipeline (processor)
inputs = processor(
    text=classes,  # List of shoe types
    images=images,  # List of images to process
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Pad inputs to the same length
    do_convert_rgb=False  # Assume images are already in RGB format
)

# Pass inputs through the model to get outputs
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)



# Extract logits for image-text similarity scores
logits_per_image = outputs.logits_per_image  # Tensor of similarity scores

# Apply softmax to logits to get class probabilities
probs = logits_per_image.softmax(dim=1)  # Probabilities for each shoe type label

# Get the highest scoring label (shoe type) for each image
predicted_type = torch.argmax(probs, dim=1)  # Tensor of predicted shoe type indices

# Convert predicted shoe type index to string label
shoes_type = classes[predicted_type.item()]  # Get the shoe type label for the predicted class

# Print the predicted shoe type label
print(f"Predicted Shoe Type: {shoes_type}")

"""## Classify material"""
# List of material classes (labels)
classes = [
    'Leather',
    'Rubber',
    'Textiles',
    'Synthetics',
    'Braided',
    'Cork',
    'Nylon',
    'Neoprene',
    'Silk',
    'Velvet',
    'Foam'
]

# Process text (classes) and images using a transformer pipeline (processor)
inputs = processor(
    text=classes,  # List of material classes
    images=images,  # List of images to process
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Pad inputs to the same length
    do_convert_rgb=False  # Assume images are already in RGB format
)

# Pass inputs through the model to get outputs
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)


# Extract logits for image-text similarity scores
logits_per_image = outputs.logits_per_image  # Tensor of similarity scores

# Apply softmax to logits to get class probabilities
probs = logits_per_image.softmax(dim=1)  # Probabilities for each material label

# Get the highest scoring label (material) for each image
predicted_material = torch.argmax(probs, dim=1)  # Tensor of predicted material indices

# Convert predicted material index to string label
material = classes[predicted_material.item()]  # Get the material label for the predicted class

# Print the predicted material label
print(f"Predicted Material: {material}")

"""## Comparing input and model output section
"""
# Broken Stitches
if damage == 'Broken stitches':
    if shoes_type == input_customer_type and input_customer_restoration == "Patch & Sewing Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Open seam
elif damage == 'Opened seam':
    if shoes_type == input_customer_type and input_customer_restoration == "Patch & Sewing Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Slanted heel
elif damage == 'Slanted heel':
    if shoes_type == input_customer_type and input_customer_restoration == "Heel Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Sole not flat
elif damage == 'Sole not flat':
    if shoes_type == input_customer_type and input_customer_restoration == "Hardware Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Peeling leather/tears
elif damage == 'Peeling leather/tears':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Alternation":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Weak cementing
elif damage == 'Weak cementing':
    if shoes_type == input_customer_type and input_customer_restoration == "Half/Full sole":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Wrinkles
elif damage == 'wrinkles':
    if shoes_type == input_customer_type and input_customer_restoration == "Hardware Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Dirty shoes
elif damage == 'dirty shoes':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Care":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

# Leather Faded
elif damage == 'Leather Faded':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Customization RE-DYE":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)
else:
    print("There are no recommendations.")

"""# Demo 2: Unmatch
## User input
"""
# Customer input
input_customer_type = "Boost"
input_customer_restoration = "Hardware Repair"

# String variable for storing outcome
damage = ""
material = ""
shoes_type = ""

"""## Image import"""
# Local directory containing the images
image_folder = r'C:\Users\Smith\Desktop\CapstoneProject\DamageImages'

# List of image filenames in the local directory
damage_image_paths = [
    'wrinkles.jpg',
]

# Load images
images = []
for filename in damage_image_paths:
    img_path = os.path.join(image_folder, filename)  # Construct the full path to the image file
    images.append(Image.open(img_path))  # Open and append the image to the list

# Assuming `image_grid` is defined earlier in the script
grid = image_grid(images, cols=3)  # Create an image grid with 3 columns
grid.show()  # Display the image grid using .show()

"""## Classify damage"""
classes = [
    'Broken stitches',
    'Opened seam',
    'Slanted heel',
    'Sole not flat',
    'Peeling leather/tears',
    'Weak cementing',
    'wrinkles',
    'dirty shoes',
    'Leather Faded'
]
inputs = processor(text=classes, images=images, return_tensors="pt", padding=True, do_convert_rgb=False)
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)

logits_per_image = outputs.logits_per_image  # This is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # Convert logits to label probabilities
# Get the highest scoring label for each image
predicted_damage = torch.argmax(probs, dim=1)
damage = str(classes[predicted_damage])
print(damage)

"""## Classify types"""
classes = [
    'Dress Shoes',
    'Loafer',
    'Boots (Ankle)',
    'Boots (Knee High)',
    'High Heels',
    'Stilettos',
    'Sandals',
    'Runners of shoes'
]
inputs = processor(text=classes, images=images, return_tensors="pt", padding=True, do_convert_rgb=False)
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)

logits_per_image = outputs.logits_per_image  # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Convert logits to label probabilities
# Get the highest scoring label for each image
predicted_type = torch.argmax(probs, dim=1)
shoes_type = str(classes[predicted_type])
print(shoes_type)

"""## Classify material"""
classes = [
    'Leather',
    'Rubber',
    'Textiles',
    'Synthetics',
    'Braided',
    'Cork',
    'Nylon',
    'Neoprene',
    'Silk',
    'Velvet',
    'Foam'
]
inputs = processor(text=classes, images=images, return_tensors="pt", padding=True, do_convert_rgb=False)
#outputs = model(inputs)
####New block
outputs = model(
    pixel_values=inputs["pixel_values"],  # Image data
    input_ids=inputs["input_ids"],        # Tokenized text data
    attention_mask=inputs["attention_mask"]  # Attention mask for text
)

logits_per_image = outputs.logits_per_image  # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Convert logits to label probabilities
# Get the highest scoring label for each image
predicted_material = torch.argmax(probs, dim=1)
material = str(classes[predicted_material])
print(material)

"""## Comparing input and model output section"""
# Comparison logic
if damage == 'Broken stitches':
    if shoes_type == input_customer_type and input_customer_restoration == "Patch & Sewing Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Opened seam':
    if shoes_type == input_customer_type and input_customer_restoration == "Patch & Sewing Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Slanted heel':
    if shoes_type == input_customer_type and input_customer_restoration == "Heel Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Sole not flat':
    if shoes_type == input_customer_type and input_customer_restoration == "Hardware Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Peeling leather/tears':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Alternation":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Weak cementing':
    if shoes_type == input_customer_type and input_customer_restoration == "Half/Full sole":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'wrinkles':
    if shoes_type == input_customer_type and input_customer_restoration == "Hardware Repair":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'dirty shoes':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Care":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

elif damage == 'Leather Faded':
    if shoes_type == input_customer_type and input_customer_restoration == "Leather Customization RE-DYE":
        print_info("correct", shoes_type, damage, material, input_customer_type, input_customer_restoration)
    else:
        print_info("wrong", shoes_type, damage, material, input_customer_type, input_customer_restoration)

else:
    print("There are no recommendations.")

