from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from neuralintents.assistants import BasicAssistant

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Message(BaseModel):
    text: str


assistant = BasicAssistant("intents.json")
assistant.fit_model(epochs=60)
assistant.save_model()


@app.get("/", response_class=HTMLResponse)
async def read_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(message: Message):
    user_message = message.text
    response = assistant.process_input(user_message)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
