from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('./models/task_classifier.pkl')
vectorizer = joblib.load('./models/vectorizer.pkl')
encoder = joblib.load('./models/encoder.pkl')

class MessageInput(BaseModel):
    message: str


app = FastAPI(title="NLP-based message routing system",description="It's an NLP-based message routing system that classifies customer requests as mobile app or web app tasks for a software company.")

# Allow CORS for all domains (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allows all domains
    allow_credentials=True,
    allow_methods=["*"],          # Allows all HTTP methods
    allow_headers=["*"],          # Allows all headers
)


@app.get('/')
def task_router_model():
    return {
        "message": "It's an NLP-based message routing system that classifies customer requests as mobile app or web app tasks for a software company."
    }

@app.post('/predict')
def predict(data: MessageInput):
    message = data.message
    
    # vectorize input message
    X_vec = vectorizer.transform([message])

    # predict
    prediction = model.predict(X_vec)

    # decode prediction class
    prediction_class = encoder.classes_[prediction[0]]
    return {"prediction": prediction_class}
