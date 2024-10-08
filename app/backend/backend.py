import logging

from ecoli import *
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulation Parameters
medium_size = (200, 200)
num_steps = 1000

# Nutrient and Toxin Fields
nutrient_field = np.zeros(medium_size)
nutrient_field[150, 150] = 500  # Initial nutrient source
toxin_field = np.zeros(medium_size)
toxin_field[50, 50] = 500  # Initial toxin source

diffusion_rate_nutrient = 0.05
diffusion_rate_toxin = 0.03
decay_rate_nutrient = 0.005
decay_rate_toxin = 0.003


app = FastAPI()
ecoli = EColi(position=[100, 100], sensing_range=5, medium_size=[200, 200])
nutrient_field = diffuse_and_decay(
    nutrient_field, diffusion_rate_nutrient, decay_rate_nutrient
)
toxin_field = diffuse_and_decay(toxin_field, diffusion_rate_toxin, decay_rate_toxin)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            ecoli.run_and_tumble(nutrient_field, toxin_field)
            await websocket.send_json({"message": ecoli.get_state()})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
