# pylint: skip-file

from fastapi import FastAPI
import uvicorn
from flows import create_flow

app = FastAPI()

@app.get("/generate_lyrics/")
async def generate_lyrics(movie_topic: str):
    soundtrack_flow = create_flow()
    return await soundtrack_flow.start(topic=movie_topic, verbose=True)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)
