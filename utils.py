from sqlmodel import SQLModel, Field, Session, create_engine, select
import requests
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine("sqlite:///db.sqlite3")

class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: str
    tokens: int = 1000

SQLModel.metadata.create_all(engine)

def get_or_create_user(email: str):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            user = User(email=email, tokens=1000)
            session.add(user)
            session.commit()
            session.refresh(user)
        return user

def deduct_tokens(email: str, amount: int):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if user and user.tokens >= amount:
            user.tokens -= amount
            session.add(user)
            session.commit()
            return True
        return False

def get_aqi(lat, lon):
    api_key = os.getenv("OPENWEATHERMAP_API")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        return response.json()['list'][0]['main']['aqi']
    except:
        return "Unavailable"
