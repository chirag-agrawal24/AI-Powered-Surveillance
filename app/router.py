from fastapi import APIRouter
# from app.api.frontend import frontend_router
from app.api.chat import chat_router
main_router = APIRouter()

# main_router.include_router(frontend_router)
main_router.include_router(chat_router)