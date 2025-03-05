from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch
import hashlib
import logging
import uvicorn  # Ensure this is imported

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define labels related to shoe damage identification
LABELS = [
    "Broken stitches", "Opened seam", "Slanted heel", "Sole not flat",
    "Peeling leather/tears", "Weak cementing", "Wrinkles",
    "Dirty shoes", "Leather faded"
]

# Define possible shoe materials for AI classification
MATERIAL_LABELS = ["Leather", "Canvas", "Suede", "Synthetic", "Rubber"]

# Define appropriate repair recommendations
REPAIR_MAPPING = {
    "Broken stitches": "Leather Care",
    "Opened seam": "Stitch Repair",
    "Slanted heel": "Heel Adjustment",
    "Sole not flat": "Sole Flattening",
    "Peeling leather/tears": "Leather Customization RE-DYE",
    "Weak cementing": "Re-Cementing",
    "Wrinkles": "Leather Customization RE-DYE",
    "Dirty shoes": "Cleaning Service",
    "Leather faded": "Leather Restoration"
}

# Define fixed metadata categories
ERA_MAPPING = ["Vintage", "Modern", "Classic", "Retro", "Antique"]
VALUE_MAPPING = ["Low", "Medium", "High", "Rare", "Collector's Item"]
CONDITION_MAPPING = {
    "Broken stitches": "Fair",
    "Opened seam": "Fair",
    "Slanted heel": "Good",
    "Sole not flat": "Poor",
    "Peeling leather/tears": "Poor",
    "Weak cementing": "Very Poor",
    "Wrinkles": "Fair",
    "Dirty shoes": "Good",
    "Leather faded": "Fair"
}

# Load AI Model
try:
    logger.info("Loading AI Model...")
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    logger.info("AI Model Loaded Successfully.")
except Exception as e:
    logger.error(f"Failed to load AI model: {str(e)}")
    model, processor = None, None  # Prevent API from running if model isn't loaded


# Define Pydantic Model for Structured API Response
class ImageAnalysisResponse(BaseModel):
    success: bool
    image_name: str
    predicted_damage: str
    recommended_repair: str
    name_of_product: str
    era: str
    material: str
    condition: str
    value: str


def compute_image_hash(image_path):
    """ Generates a stable hash based on the image content."""
    hasher = hashlib.sha256()
    with open(image_path, 'rb') as img_file:
        while chunk := img_file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_hash(hash_str, choices):
    """ Uses a consistent hash from image content to determine metadata values."""
    hash_val = int(hash_str, 16)
    return choices[hash_val % len(choices)]


def predict_material(image_path):
    """ Uses AI to predict the shoe material accurately."""
    try:
        if model is None or processor is None:
            raise RuntimeError("AI Model is not loaded.")

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=MATERIAL_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask")
            )
        probs = outputs.logits_per_image.softmax(dim=1)  # Convert logits to probabilities
        predicted_idx = torch.argmax(probs, dim=1).item()
        return MATERIAL_LABELS[predicted_idx]
    except Exception as e:
        logger.error(f"Material Prediction Error: {str(e)}")
        return "Unknown"


def process_image(image_path):
    """
    Uses AI to analyze the shoe image, predict damages and material, and return accurate repair recommendations.
    """
    try:
        if model is None or processor is None:
            raise RuntimeError("AI Model is not loaded.")

        # Compute a stable hash based on the image content
        image_hash = compute_image_hash(image_path)
        logger.info(f"Processing image with hash: {image_hash[:10]}")

        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Run AI model for damage prediction
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask")
            )

        probs = outputs.logits_per_image.softmax(dim=1)  # Convert logits to probabilities
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_label = LABELS[predicted_idx]
        repair_suggestion = REPAIR_MAPPING.get(predicted_label, "Unknown Repair Suggestion")

        # AI-based shoe material classification
        predicted_material = predict_material(image_path)

        # Stable metadata generation based on image content hash
        return ImageAnalysisResponse(
            success=True,
            image_name=os.path.basename(image_path),
            predicted_damage=predicted_label,
            recommended_repair=repair_suggestion,
            name_of_product=f"Product-{int(image_hash[:8], 16) % 10000}",
            era=stable_hash(image_hash, ERA_MAPPING),
            material=predicted_material,  # AI-determined material
            condition=CONDITION_MAPPING.get(predicted_label, "Unknown"),
            value=stable_hash(image_hash, VALUE_MAPPING)
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/process-image/", response_model=ImageAnalysisResponse)
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload and process an image, returning AI-generated shoe damage and material analysis.
    """
    try:
        if model is None or processor is None:
            raise HTTPException(status_code=500, detail="AI Model is not loaded.")

        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process image using AI model
        result = process_image(file_path)

        # Clean up
        os.remove(file_path)

        return JSONResponse(content=result.dict(), headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# **FIXED PORT BINDING**
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
