import aiofiles, os, uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.responses import JSONResponse
from ultralytics import YOLO

from apiUtils.Hasher import HasherClass
from pathlib import Path

# Load YOLO model.
model = YOLO('1st_best_of_100ep.pt')

# Create necessary directories if they don't exist.
content_dir = os.path.join(os.getcwd(), "api", "Content")
predict_images_dir = os.path.join(content_dir, "Predict_Images")

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
@app.post('/test')
async def test(upload_image: UploadFile):
    """
    :param upload_image: File of photo that contains ipu and QR code on it
    :return: JSON response containing predicted values
    """

    try:
        # Save uploaded image.
        hashedFileName = HasherObject.CreateImageFileNameHash(upload_image.filename)
        async with aiofiles.open((Path() / "Content" / "Predict_Images" / hashedFileName).absolute(),
                                 'wb') as image_file:
            await image_file.write(await upload_image.read())

        # Log image save success.
        logger.debug(f"Image saved: {hashedFileName}")

        # Make prediction using YOLO model.
        prediction_class = model.predict((Path() / "Content" / "Predict_Images" / hashedFileName).absolute(), conf=0.6)

        # Extract prediction details.
        confidence = prediction_class[0].probs.top1conf.item()
        name = prediction_class[0].names[prediction_class[0].probs.top1]
        series = ""
        number = ""

        # Extract additional details based on prediction.
        if name == "vehicle_passport":
            type = name
            page_number = 0
        else:
            type = name.split("-")[0]
            page_number = name.split("-")[1]

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

        # prediction_detect = detect.prediction('v2.pt', 'data.yaml',
        #                                (Path() / "Content" / "Predict_Images" / hashedFileName).absolute())

        if prediction_class is None:
            raise HTTPException(status_code=404, detail='Bad Class')
        # if prediction_detect is None:
        #     raise HTTPException(status_code=417, detail='Bad Detect')

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
