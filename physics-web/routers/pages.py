from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from data.content import (
    STATS, RESEARCH_AREAS, FACULTY, NEWS, TIMELINE, CONTACT_INFO
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": STATS,
            "research_areas": RESEARCH_AREAS,
            "faculty_preview": FACULTY[:4],
            "news": NEWS[:6],
            "page": "home",
        },
    )


@router.get("/about")
async def about(request: Request):
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "timeline": TIMELINE,
            "stats": STATS,
            "page": "about",
        },
    )


@router.get("/faculty")
async def faculty(request: Request):
    specialties = list(dict.fromkeys(f["specialty"] for f in FACULTY))
    return templates.TemplateResponse(
        "faculty.html",
        {
            "request": request,
            "faculty": FACULTY,
            "specialties": specialties,
            "page": "faculty",
        },
    )


@router.get("/research")
async def research(request: Request):
    return templates.TemplateResponse(
        "research.html",
        {
            "request": request,
            "research_areas": RESEARCH_AREAS,
            "page": "research",
        },
    )


@router.get("/contact")
async def contact(request: Request):
    return templates.TemplateResponse(
        "contact.html",
        {
            "request": request,
            "contact": CONTACT_INFO,
            "page": "contact",
        },
    )
