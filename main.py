import shutil

import aiofiles, os, uvicorn
import cv2
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

import CRNN
from apiUtils.Hasher import HasherClass
from pathlib import Path
import numpy as np

# Load YOLO model.
model = YOLO('2nd_best_50ep.pt')
model2 = YOLO('best_detect.onnx')

# Create necessary directories if they don't exist.
content_dir = os.path.join(os.getcwd(), "api", "Content")
predict_images_dir = os.path.join(content_dir, "Predict_Images")

tmp_dir = os.path.join(os.getcwd(), "tmp")
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

if not os.path.exists(content_dir):
    os.makedirs(content_dir)

if not os.path.exists(predict_images_dir):
    os.makedirs(predict_images_dir)

app = FastAPI()

# Mount static files' directory.
app.mount("/static/", StaticFiles(directory="Content"), name="static")
logger.remove()
# Add logging configuration.
logger.add('debug.log', format="{time} {message}", level="DEBUG", rotation="2 MB", compression="zip")
HasherObject = HasherClass()

origins = [
    "*"
]

# Define CORS settings.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@logger.catch
@app.post('/detect')
async def detect(upload_image: UploadFile):
    """
    :param upload_image: File of photo that contains ipu and QR code on it
    :return: JSON response containing predicted values
    """

    try:
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # Save uploaded image.
        hashedFileName = HasherObject.CreateImageFileNameHash(upload_image.filename)
        async with aiofiles.open((Path() / "Content" / "Predict_Images" / hashedFileName).absolute(),
                                 'wb') as image_file:
            await image_file.write(await upload_image.read())

        # Log image save success.
        logger.debug(f"Image saved: {hashedFileName}")

        # Make prediction using YOLO model.
        prediction_class = model.predict((Path() / "Content" / "Predict_Images" / hashedFileName).absolute(), conf=0.6)

        if prediction_class[0].probs is None:
            raise HTTPException(status_code=404, detail='Bad Class')

        # Extract prediction details.
        confidence = prediction_class[0].probs.top1conf.item()
        name = prediction_class[0].names[prediction_class[0].probs.top1]
        print(name)
        if name in ["vehicle_passport", "Unknown"]:
            type = name
            page_number = 0
        else:
            type = name.split("-")[0]
            page_number = name.split("-")[1]

        # Log prediction details.
        logger.debug(f"Prediction: Type - {type}, Confidence - {confidence:.4f}, Page Number - {page_number}")

        prediction_detect = model2.predict((Path() / "Content" / "Predict_Images" / hashedFileName).absolute(),
                                           conf=0.5)[0]

        image = Image.open((Path() / "Content" / "Predict_Images" / hashedFileName).absolute())
        boxes = prediction_detect.boxes

        print(len(boxes))
        if boxes is None or len(boxes) == 0:
            return {
                "type": type,
                "confidence": round(confidence, 4),
                "series": "",
                "number": "",
                "page_number": page_number
            }

        max_box = max(boxes, key=lambda x: float(x.conf[0]))[0]

        cropped_image = image.crop(max_box.xyxy.tolist()[0])
        if cropped_image.size[0] < cropped_image.size[1]:
            cropped_image = cropped_image.rotate(-270, expand=True)
        cropped_image = cropped_image.resize((250, 50))
        hashedFileName = HasherObject.CreateImageFileNameHash(upload_image.filename)
        cropped_image_path = os.path.join(tmp_dir, "cropped_" + hashedFileName)

        cropped_image.save(cropped_image_path)

        detected = CRNN.predict(tmp_dir)
        series = detected[:4]
        number = detected[4:]

        # Log prediction details.
        logger.debug(f"Prediction: Type - {type}, Confidence - {confidence:.4f}, Page Number - {page_number}")

        # Create dictionary containing all values.
        all_values = {
            "type": type,
            "confidence": round(confidence, 4),
            "series": series,
            "number": number,
            "page_number": page_number
        }
        print(all_values)
        return JSONResponse(all_values)

    # Raise exception if there's an IndexError.
    except IndexError:
        raise HTTPException(status_code=401, detail='Bad APIToken')


# --------------------------------------------------------------------------

if __name__ == '__main__':
    uvicorn.run("main:app",
                host="localhost",
                port=5500,
                reload=True
                )
