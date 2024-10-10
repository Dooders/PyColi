import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict

import numpy as np
from ecoli import EColi
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app, Summary
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

# Load configuration from environment variables
MEDIUM_SIZE = tuple(map(int, os.getenv("MEDIUM_SIZE", "200,200").split(",")))
NUM_STEPS = int(os.getenv("NUM_STEPS", "1000"))
DIFFUSION_RATE_NUTRIENT = float(os.getenv("DIFFUSION_RATE_NUTRIENT", "0.05"))
DIFFUSION_RATE_TOXIN = float(os.getenv("DIFFUSION_RATE_TOXIN", "0.03"))
DECAY_RATE_NUTRIENT = float(os.getenv("DECAY_RATE_NUTRIENT", "0.005"))
DECAY_RATE_TOXIN = float(os.getenv("DECAY_RATE_TOXIN", "0.003"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Configure logging
from logging import config as logging_config

logging_config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Nutrient and Toxin Fields
nutrient_field = np.zeros(MEDIUM_SIZE)
toxin_field = np.zeros(MEDIUM_SIZE)

# Global variables for metrics
metrics = None
metrics_initialized = False
metrics_lock = Lock()

# Add these global variables
SIMULATION_DURATION = Summary('simulation_duration_seconds', 'Time spent in simulation step')
WEBSOCKET_CONNECTIONS = Counter('websocket_connections_total', 'Total WebSocket connections')


def create_metrics():
    global metrics, metrics_initialized
    with metrics_lock:
        if not metrics_initialized:
            registry = CollectorRegistry()
            metrics = {
                "REQUESTS": Counter(
                    "http_requests_total", "Total HTTP Requests", registry=registry
                ),
                "WEBSOCKET_CONNECTIONS": Counter(
                    "websocket_connections_total",
                    "Total WebSocket Connections",
                    registry=registry,
                ),
                "SIMULATION_DURATION": Histogram(
                    "simulation_duration_seconds",
                    "Simulation Duration in Seconds",
                    registry=registry,
                ),
            }
            metrics_initialized = True
    return metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize metrics here
    create_metrics()
    # Expose Prometheus metrics
    instrumentator.expose(app)
    yield
    logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)

# Add Prometheus metrics
metrics_app = make_asgi_app(registry=CollectorRegistry())
app.mount("/metrics", metrics_app)

instrumentator = Instrumentator().instrument(app)

ecoli = None
simulation_running = False
simulation_step = 0

# Add a dictionary to store client-specific simulations
client_simulations = {}


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Implement rate limiting logic here
        return await call_next(request)


# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# Add middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_fields():
    nutrient_field = np.zeros(MEDIUM_SIZE)
    toxin_field = np.zeros(MEDIUM_SIZE)

    # Add some initial nutrients
    center = (MEDIUM_SIZE[0] // 2, MEDIUM_SIZE[1] // 2)
    radius = min(MEDIUM_SIZE) // 4

    for i in range(MEDIUM_SIZE[0]):
        for j in range(MEDIUM_SIZE[1]):
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


async def get_client_simulation(websocket: WebSocket) -> Dict[str, Any]:
    client_id = id(websocket)
    if client_id not in client_simulations:
        raise HTTPException(status_code=404, detail="Client simulation not found")
    return client_simulations[client_id]


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    WEBSOCKET_CONNECTIONS.inc()

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
            logger.info(f"Received command: {command} from client {client_id}")

            simulation = await get_client_simulation(websocket)

            if command == "start":
                simulation["simulation_running"] = True
                if simulation["ecoli"] is None:
                    simulation["ecoli"] = EColi(
                        position=[100, 100], sensing_range=5, medium_size=MEDIUM_SIZE
                    )
                    (
                        simulation["nutrient_field"],
                        simulation["toxin_field"],
                    ) = initialize_fields()
                logger.info(f"Simulation started for client {client_id}")

            elif command == "pause":
                simulation["simulation_running"] = False
                logger.info(f"Simulation paused for client {client_id}")

            elif command == "restart":
                simulation["simulation_running"] = True
                simulation["simulation_step"] = 0
                simulation["ecoli"] = EColi(
                    position=[100, 100], sensing_range=5, medium_size=MEDIUM_SIZE
                )
                (
                    simulation["nutrient_field"],
                    simulation["toxin_field"],
                ) = initialize_fields()
                logger.info(f"Simulation restarted for client {client_id}")

            else:
                logger.warning(f"Unknown command received: {command} from client {client_id}")
                await websocket.send_json({"error": "Unknown command"})
                continue

            while simulation["simulation_running"]:
                start_time = time.time()

                ecoli = simulation["ecoli"]
                nutrient_field = simulation["nutrient_field"]
                toxin_field = simulation["toxin_field"]
                ecoli_state = ecoli.get_state()
                ecoli_state["cycle"] = simulation["simulation_step"]

                # Add logging of E. coli state
                logger.info(
                    f"E. coli state (step {simulation['simulation_step']}, client {client_id}): {ecoli_state}"
                )

                nutrient_field = diffuse_and_decay(
                    nutrient_field, DIFFUSION_RATE_NUTRIENT, DECAY_RATE_NUTRIENT
                )
                toxin_field = diffuse_and_decay(
                    toxin_field, DIFFUSION_RATE_TOXIN, DECAY_RATE_TOXIN
                )

                # Add logging to check field values
                logger.info(f"Nutrient field sum: {np.sum(nutrient_field)}, client {client_id}")
                logger.info(f"Toxin field sum: {np.sum(toxin_field)}, client {client_id}")

                data_to_send = {
                    "ecoli_state": ecoli_state,
                    "nutrient_field": nutrient_field.tolist(),
                    "toxin_field": toxin_field.tolist(),
                }

                # Log the size of the data being sent
                logger.info(f"Data size: {len(str(data_to_send))} bytes, client {client_id}")

                await websocket.send_json(data_to_send)

                simulation["simulation_step"] += 1
                await asyncio.sleep(0.1)

                simulation_duration = time.time() - start_time
                SIMULATION_DURATION.observe(simulation_duration)
                logger.info(f"Simulation step duration: {simulation_duration:.4f} seconds, client {client_id}")

                # Check for new messages
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=0.01
                    )
                    command = message.get("command")
                    if command == "pause":
                        simulation["simulation_running"] = False
                        logger.info(f"Simulation paused for client {client_id}")
                        break
                    elif command == "restart":
                        simulation["simulation_running"] = True
                        simulation["simulation_step"] = 0
                        simulation["ecoli"] = EColi(
                            position=[100, 100],
                            sensing_range=5,
                            medium_size=MEDIUM_SIZE,
                        )
                        (
                            simulation["nutrient_field"],
                            simulation["toxin_field"],
                        ) = initialize_fields()
                        logger.info(f"Simulation restarted for client {client_id}")
                        break
                except asyncio.TimeoutError:
                    pass  # No new message, continue simulation

            if not simulation["simulation_running"]:
                await asyncio.sleep(
                    0.1
                )  # Add a small delay when not running to avoid busy-waiting

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
    except Exception as e:
        logger.exception(f"Error in WebSocket connection for client {client_id}: {str(e)}")
    finally:
        if client_id in client_simulations:
            del client_simulations[client_id]
        logger.info(f"Cleaned up simulation data for client {client_id}")


@app.get("/health")
async def health_check():
    # Add more detailed health checks here
    return {
        "status": "healthy",
        "active_simulations": len(client_simulations),
        "websocket_connections": WEBSOCKET_CONNECTIONS._value.get(),
    }


@app.get("/ready")
async def readiness_check():
    # Add more detailed readiness checks here
    return {
        "status": "ready",
        "active_simulations": len(client_simulations),
        "websocket_connections": WEBSOCKET_CONNECTIONS._value.get(),
    }


# Add a new endpoint for simulation statistics
@app.get("/stats")
async def simulation_stats():
    return {
        "active_simulations": len(client_simulations),
        "websocket_connections": WEBSOCKET_CONNECTIONS._value.get(),
        "avg_simulation_duration": SIMULATION_DURATION._sum.get() / SIMULATION_DURATION._count.get() if SIMULATION_DURATION._count.get() > 0 else 0,
    }


if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import Config

    logger.info("Starting the application")
    config = Config(
        "backend:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_config="logging.conf",
        loop="uvloop",
        http="httptools",
    )
    server = uvicorn.Server(config)
    server.run()