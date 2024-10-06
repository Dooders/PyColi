// Create a WebSocket connection to the FastAPI backend
let socket;
let simulationRunning = false;

function connectWebSocket() {
  socket = new WebSocket("ws://localhost:8000/ws/simulation");

  // WebSocket event listener for receiving simulation data
  socket.onmessage = function (event) {
    const data = JSON.parse(event.data);

    // Update E. coli position
    const ecoliPosition = data.ecoli_position;
    ecoliMesh.position.set(ecoliPosition[1], ecoliPosition[0], 0);

    // Optionally update the nutrient and toxin fields visualization
    updateFields(data.nutrient_field, data.toxin_field);
  };
}

// Three.js setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create a plane to represent the medium
tempGeometry = new THREE.PlaneGeometry(200, 200);
tempMaterial = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, side: THREE.DoubleSide });
const plane = new THREE.Mesh(tempGeometry, tempMaterial);
scene.add(plane);

// Create the E. coli mesh
const ecoliGeometry = new THREE.SphereGeometry(1, 32, 32);
const ecoliMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
const ecoliMesh = new THREE.Mesh(ecoliGeometry, ecoliMaterial);
scene.add(ecoliMesh);

// Create a texture to visualize the nutrient and toxin fields
const fieldSize = 200;
const fieldCanvas = document.createElement("canvas");
fieldCanvas.width = fieldSize;
fieldCanvas.height = fieldSize;
const fieldContext = fieldCanvas.getContext("2d");
const fieldTexture = new THREE.CanvasTexture(fieldCanvas);
const fieldMaterial = new THREE.MeshBasicMaterial({ map: fieldTexture, transparent: true });
const fieldMesh = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), fieldMaterial);
scene.add(fieldMesh);

camera.position.z = 300;

function animate() {
  if (simulationRunning) {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
}

// Function to update nutrient and toxin fields
function updateFields(nutrientField, toxinField) {
  const imageData = fieldContext.createImageData(fieldSize, fieldSize);
  for (let x = 0; x < fieldSize; x++) {
    for (let y = 0; y < fieldSize; y++) {
      const index = (y * fieldSize + x) * 4;
      const nutrientValue = nutrientField[y][x];
      const toxinValue = toxinField[y][x];

      // Scale nutrient and toxin values to be between 0 and 255
      const nutrientIntensity = Math.min(255, nutrientValue);
      const toxinIntensity = Math.min(255, toxinValue);

      // Set the color values for the nutrient and toxin fields
      imageData.data[index] = toxinIntensity; // Red channel for toxins
      imageData.data[index + 1] = nutrientIntensity; // Green channel for nutrients
      imageData.data[index + 2] = 0; // Blue channel
      imageData.data[index + 3] = 255; // Alpha channel
    }
  }
  fieldContext.putImageData(imageData, 0, 0);
  fieldTexture.needsUpdate = true;
}

// Handle window resizing
window.addEventListener("resize", function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// UI Controls for Simulation
const controlsDiv = document.createElement("div");
controlsDiv.style.position = "absolute";
controlsDiv.style.top = "10px";
controlsDiv.style.left = "10px";
controlsDiv.style.backgroundColor = "rgba(255, 255, 255, 0.8)";
controlsDiv.style.padding = "10px";
controlsDiv.style.borderRadius = "5px";
document.body.appendChild(controlsDiv);

const startButton = document.createElement("button");
startButton.innerText = "Start Simulation";
startButton.onclick = function () {
  if (!simulationRunning) {
    simulationRunning = true;
    connectWebSocket();
    animate();
  }
};
controlsDiv.appendChild(startButton);

const pauseButton = document.createElement("button");
pauseButton.innerText = "Pause Simulation";
pauseButton.onclick = function () {
  simulationRunning = false;
  if (socket) {
    socket.close();
  }
};
controlsDiv.appendChild(pauseButton);

const restartButton = document.createElement("button");
restartButton.innerText = "Restart Simulation";
restartButton.onclick = function () {
  if (socket) {
    socket.close();
  }
  simulationRunning = true;
  connectWebSocket();
  animate();
};
controlsDiv.appendChild(restartButton);