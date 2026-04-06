from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from routers.pages import router as pages_router

app = FastAPI(
    title="부산대학교 물리학과",
    description="Pusan National University - Department of Physics",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(pages_router)


@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/images/favicon.ico")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
