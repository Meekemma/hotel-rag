from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Grand Lekki Hotel Assistant")
app.include_router(router)
