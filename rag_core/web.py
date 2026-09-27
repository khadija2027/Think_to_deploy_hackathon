"""One web application, with no accounts, LDAP, JWT, or session cookies."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from .service import GenerationUnavailable, IndexNotReady, RAGService


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        if not value.strip():
            raise ValueError("La question ne peut pas être vide.")
        return value.strip()


def create_app(service=None):
    app = FastAPI(title="Assistant RH — RAG", version="2.0.0")
    app.state.rag = service or RAGService()
    frontend = Path(__file__).resolve().parents[1] / "frontend_chatbot"
    templates = Jinja2Templates(directory=str(frontend / "templates"))
    app.mount("/static", StaticFiles(directory=str(frontend / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/chatbot", response_class=HTMLResponse)
    def chatbot(request: Request):
        return templates.TemplateResponse(request=request, name="chatbot.html", context={})

    @app.post("/api/ask")
    def ask(payload: Question):
        try:
            return app.state.rag.ask(payload.question)
        except IndexNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GenerationUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "rag": app.state.rag.status()}

    @app.get("/api/status")
    def status():
        return app.state.rag.status()

    @app.get("/ready")
    def ready():
        state = app.state.rag.status()
        return JSONResponse(state, status_code=200 if state["ready"] else 503)

    return app
