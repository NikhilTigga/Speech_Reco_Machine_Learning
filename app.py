from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import librosa
import numpy as np
import pickle
import os
import logging
from fastapi.staticfiles import StaticFiles


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the pre-trained SVM model and the scaler
try:
    with open('svm_model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Function to extract features from audio files
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zero_crossing_rate.T, axis=0)
    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse.T, axis=0)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, zcr_mean, rmse_mean, tonnetz_mean])
    return features

# Function to predict new audio
def predict_audio(audio_path):
    features = extract_features(audio_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify-audio")
async def classify_audio(audio: UploadFile = File(...)):
    if audio.content_type not in ["audio/wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV or MP3 file.")

    audio_path = 'temp_audio_file.wav'
    try:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        prediction = predict_audio(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}")
    finally:
        os.remove(audio_path)

    return JSONResponse(content={'prediction': prediction})

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
