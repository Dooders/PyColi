import asyncio
import logging

import numpy as np
from ecoli import EColi
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
toxin_field = np.zeros(medium_size)

diffusion_rate_nutrient = 0.05
diffusion_rate_toxin = 0.03
decay_rate_nutrient = 0.005
decay_rate_toxin = 0.003

app = FastAPI()
ecoli = None
simulation_running = False
simulation_step = 0

# Add a dictionary to store client-specific simulations
client_simulations = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def initialize_fields():
    nutrient_field = np.zeros(medium_size)
    toxin_field = np.zeros(medium_size)

    # Add some initial nutrients
    center = (medium_size[0] // 2, medium_size[1] // 2)
    radius = min(medium_size) // 4

    for i in range(medium_size[0]):
        for j in range(medium_size[1]):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance < radius:
                nutrient_field[i, j] = 1.0 - (distance / radius)

    # Add initial toxin
    toxin_field[50, 50] = 1.0

    # Log the initial field values
    logger.info(f"Initial nutrient field sum: {np.sum(nutrient_field)}")
    logger.info(f"Initial toxin field sum: {np.sum(toxin_field)}")

    return nutrient_field, toxin_field


def diffuse_and_decay(field, diffusion_rate, decay_rate):
    new_field = field.copy()
    for i in range(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            new_field[i, j] += diffusion_rate * (
                field[i - 1, j]
                + field[i + 1, j]
                + field[i, j - 1]
                + field[i, j + 1]
                - 4 * field[i, j]
            )
    new_field *= 1 - decay_rate
    return new_field


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    global client_simulations
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Generate a unique client ID
    client_id = id(websocket)
    client_simulations[client_id] = {
        "ecoli": None,
        "nutrient_field": None,
        "toxin_field": None,
        "simulation_running": False,
        "simulation_step": 0,
    }

    try:
        while True:
            message = await websocket.receive_json()
            command = message.get("command")
            logger.info(f"Received command: {command}")

            if command == "start":
                client_simulations[client_id]["simulation_running"] = True
                if client_simulations[client_id]["ecoli"] is None:
                    client_simulations[client_id]["ecoli"] = EColi(
                        position=[100, 100], sensing_range=5, medium_size=medium_size
                    )
                    (
                        client_simulations[client_id]["nutrient_field"],
                        client_simulations[client_id]["toxin_field"],
                    ) = initialize_fields()
                logger.info(f"Simulation started for client {client_id}")

            elif command == "pause":
                client_simulations[client_id]["simulation_running"] = False
                logger.info(f"Simulation paused for client {client_id}")

            elif command == "restart":
                client_simulations[client_id]["simulation_running"] = True
                client_simulations[client_id]["simulation_step"] = 0
                client_simulations[client_id]["ecoli"] = EColi(
                    position=[100, 100], sensing_range=5, medium_size=medium_size
                )
                (
                    client_simulations[client_id]["nutrient_field"],
                    client_simulations[client_id]["toxin_field"],
                ) = initialize_fields()
                logger.info(f"Simulation restarted for client {client_id}")

            while client_simulations[client_id]["simulation_running"]:
                ecoli = client_simulations[client_id]["ecoli"]
                nutrient_field = client_simulations[client_id]["nutrient_field"]
                toxin_field = client_simulations[client_id]["toxin_field"]
                ecoli_state = ecoli.get_state()
                ecoli_state["cycle"] = client_simulations[client_id]["simulation_step"]

                # Add logging of E. coli state
                logger.info(
                    f"E. coli state (step {client_simulations[client_id]['simulation_step']}): {ecoli_state}"
                )

                nutrient_field = diffuse_and_decay(
                    nutrient_field, diffusion_rate_nutrient, decay_rate_nutrient
                )
                toxin_field = diffuse_and_decay(
                    toxin_field, diffusion_rate_toxin, decay_rate_toxin
                )

                # Add logging to check field values
                logger.info(f"Nutrient field sum: {np.sum(nutrient_field)}")
                logger.info(f"Toxin field sum: {np.sum(toxin_field)}")

                data_to_send = {
                    "ecoli_state": ecoli_state,
                    "nutrient_field": nutrient_field.tolist(),
                    "toxin_field": toxin_field.tolist(),
                }

                # Log the size of the data being sent
                logger.info(f"Data size: {len(str(data_to_send))} bytes")

                await websocket.send_json(data_to_send)

                client_simulations[client_id]["simulation_step"] += 1
                await asyncio.sleep(0.1)

                # Check for new messages
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=0.01
                    )
                    command = message.get("command")
                    if command == "pause":
                        client_simulations[client_id]["simulation_running"] = False
                        logger.info(f"Simulation paused for client {client_id}")
                        break
                    elif command == "restart":
                        client_simulations[client_id]["simulation_running"] = True
                        client_simulations[client_id]["simulation_step"] = 0
                        client_simulations[client_id]["ecoli"] = EColi(
                            position=[100, 100],
                            sensing_range=5,
                            medium_size=medium_size,
                        )
                        (
                            client_simulations[client_id]["nutrient_field"],
                            client_simulations[client_id]["toxin_field"],
                        ) = initialize_fields()
                        logger.info(f"Simulation restarted for client {client_id}")
                        break
                except asyncio.TimeoutError:
                    pass  # No new message, continue simulation

            if not client_simulations[client_id]["simulation_running"]:
                await asyncio.sleep(
                    0.1
                )  # Add a small delay when not running to avoid busy-waiting

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection for client {client_id}: {str(e)}")
    finally:
        # Clean up the client's simulation data when the connection is closed
        if client_id in client_simulations:
            del client_simulations[client_id]
        logger.info(f"Cleaned up simulation data for client {client_id}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
