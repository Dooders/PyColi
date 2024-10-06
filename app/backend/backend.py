from backend.ecoli import EColi, diffuse_and_decay
from fastapi import FastAPI, WebSocket
import numpy as np
import random
import math

app = FastAPI()

medium_size = (200, 200)
diffusion_rate_nutrient = 0.05
diffusion_rate_toxin = 0.03
decay_rate_nutrient = 0.005
decay_rate_toxin = 0.003

# Nutrient and Toxin Fields
nutrient_field = np.zeros(medium_size)
nutrient_field[150, 150] = 500
toxin_field = np.zeros(medium_size)
toxin_field[50, 50] = 500

# Initialize E. coli
ecoli = EColi(position=[100, 100], sensing_range=5, medium_size=medium_size)

def step_simulation():
    global nutrient_field, toxin_field, ecoli

    nutrient_field = diffuse_and_decay(nutrient_field, diffusion_rate_nutrient, decay_rate_nutrient)
    toxin_field = diffuse_and_decay(toxin_field, diffusion_rate_toxin, decay_rate_toxin)
    ecoli.run_and_tumble(nutrient_field, toxin_field)
    return {
        "ecoli_position": ecoli.position.tolist(),
        "nutrient_field": nutrient_field.tolist(),
        "toxin_field": toxin_field.tolist(),
    }

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = step_simulation()
            await websocket.send_json(data)
    except Exception as e:
        print(f"Connection closed: {e}")
