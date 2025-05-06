from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path

from app.config import ServerConfig

frontend_router = APIRouter()


# Mount static files directory
frontend_router.mount("/app/static", StaticFiles(directory=Path("app/static")), name="static")

# Jinja2 templates directory
templates = Jinja2Templates(directory=ServerConfig.TEMPLATES_DIR)


# ----------- ROUTES -----------

# Home page
@frontend_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# # Chat page
# @frontend_router.get("/chat", response_class=HTMLResponse)
# async def chat_page(request: Request):
#     return templates.TemplateResponse("chat.html", {"request": request})


# # Real-time detection page
# @frontend_router.get("/real-time-detection", response_class=HTMLResponse)
# async def real_time_detection(request: Request):
#     return templates.TemplateResponse("real_time_detection.html", {"request": request})


# # Batch processing page
# @frontend_router.get("/batch-process", response_class=HTMLResponse)
# async def batch_page(request: Request):
#     return templates.TemplateResponse("batch_process.html", {"request": request})
