from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware

import tensorflow as tf
import keras
import numpy as np
from PIL import Image

import os
import shutil

from utils import get_or_create_user, deduct_tokens, get_aqi

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

config = Config('.env')
oauth = OAuth()


oauth.register(
    name='google',
    client_id=config('GOOGLE_CLIENT_ID'),
    client_secret=config('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

app.add_middleware(SessionMiddleware, secret_key=config('SECRET_KEY'))

model = keras.models.load_model("model/garbage_classifier.h5")
class_names = [
    'battery',
    'biological',
    'brown-glass',
    'cardboard',
    'clothes',
    'green-glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash',
    'white-glass'
]

def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img)
    pred_idx = np.argmax(predictions)
    return class_names[pred_idx]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = request.session.get('user')
    email = user.get('email') if user else None
    user_db = get_or_create_user(email) if email else None
    tokens = user_db.tokens if user_db else None
    aqi = get_aqi(27.7172, 85.3240)
    return templates.TemplateResponse("home.html", {"request": request, "user": user, "tokens": tokens, "aqi": aqi})

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    resp = await oauth.google.get('https://openidconnect.googleapis.com/v1/userinfo', token=token)
    user_info = resp.json()
    request.session['user'] = dict(user_info)
    return RedirectResponse('/')

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse('/')

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    user = request.session.get('user')
    if not user:
        return RedirectResponse('/login')

    email = user.get('email')
    if not deduct_tokens(email, 3):
        return templates.TemplateResponse("predict.html", {"request": request, "result": "Not enough tokens", "user": user, "tokens": get_or_create_user(email).tokens})

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)
    os.remove(file_path)

    tokens = get_or_create_user(email).tokens
    return templates.TemplateResponse("predict.html", {"request": request, "result": result, "user": user, "tokens": tokens})
