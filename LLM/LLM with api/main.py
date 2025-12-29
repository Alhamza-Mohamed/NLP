from fastapi import FastAPI
import ollama

app = FastAPI()

#end point

@app.post("/generate")
def generate (prompt: str):
    response = ollama.chat(model = "mistral", messages = [{"role":"user", "content":prompt}])
    return {"response": response["message"]["content"]}