let scene, camera, renderer;
let simulationRunning = false;
let socket;

document.addEventListener("DOMContentLoaded", (event) => {
  console.log("DOM fully loaded and parsed");
  initThreeJS();
  setupUI();
  connectWebSocket();
});

function initThreeJS() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0xaaaaaa); // Set the background color to grey
  document
    .getElementById("simulation-container")
    .appendChild(renderer.domElement);

  // Create a plane to represent the medium
  const planeGeometry = new THREE.PlaneGeometry(200, 200);
  const planeMaterial = new THREE.MeshBasicMaterial({
    color: 0xaaaaaa,
    side: THREE.DoubleSide,
  });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);
  plane.name = "fieldMesh"; // Name the plane for easy reference
  scene.add(plane);

  // Create the E. coli mesh
  const ecoliGeometry = new THREE.SphereGeometry(1, 32, 32);
  const ecoliMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
  const ecoliMesh = new THREE.Mesh(ecoliGeometry, ecoliMaterial);
  ecoliMesh.name = "ecoliMesh";
  scene.add(ecoliMesh);

  // Set up the camera
  camera.position.z = 150; // Adjust this value to fit the entire plane in view

  animate();
}

function connectWebSocket() {
  console.log("Attempting to connect WebSocket");
  socket = new WebSocket("ws://localhost:8000/ws/simulation");

  socket.onopen = function (event) {
    console.log("WebSocket connection established");
  };

  socket.onmessage = function (event) {
    const data = JSON.parse(event.data);
    console.log("Received data from server:");
    console.log("E. coli state:", data.ecoli_state);
    console.log(
      "Nutrient field (first few values):",
      data.nutrient_field.slice(0, 5).map((row) => row.slice(0, 5))
    );
    console.log(
      "Toxin field (first few values):",
      data.toxin_field.slice(0, 5).map((row) => row.slice(0, 5))
    );
    updateSimulation(data);
  };

  socket.onclose = function (event) {
    console.log("WebSocket connection closed");
    setTimeout(connectWebSocket, 1000); // Attempt to reconnect after 1 second
  };

  socket.onerror = function (error) {
    console.error("WebSocket error:", error);
  };
}

function updateSimulation(data) {
  // Update E. coli position
  const ecoliMesh = scene.getObjectByName("ecoliMesh");
  if (ecoliMesh) {
    ecoliMesh.position.set(
      data.ecoli_state.position[0],
      data.ecoli_state.position[1],
      0
    );
  }

  // Check if nutrient_field and toxin_field are not all zeros
  const nutrientSum = data.nutrient_field.flat().reduce((a, b) => a + b, 0);
  const toxinSum = data.toxin_field.flat().reduce((a, b) => a + b, 0);

  console.log("Nutrient field sum:", nutrientSum);
  console.log("Toxin field sum:", toxinSum);

  if (nutrientSum === 0 && toxinSum === 0) {
    console.warn("Both nutrient and toxin fields are all zeros!");
  }

  // Update nutrient and toxin fields
  updateFields(data.nutrient_field, data.toxin_field);
}

function updateFields(nutrientField, toxinField) {
  const fieldSize = 200;
  const fieldMesh = scene.getObjectByName("fieldMesh");
  if (!fieldMesh) {
    console.error("fieldMesh not found");
    return;
  }

  // Create a new canvas texture if it doesn't exist
  if (!fieldMesh.material.map) {
    const fieldCanvas = document.createElement("canvas");
    fieldCanvas.width = fieldSize;
    fieldCanvas.height = fieldSize;
    const fieldTexture = new THREE.CanvasTexture(fieldCanvas);
    fieldMesh.material.map = fieldTexture;
  }

  const fieldTexture = fieldMesh.material.map;
  const fieldCanvas = fieldTexture.image;
  const fieldContext = fieldCanvas.getContext("2d");
  const imageData = fieldContext.createImageData(fieldSize, fieldSize);

  for (let x = 0; x < fieldSize; x++) {
    for (let y = 0; y < fieldSize; y++) {
      const index = (y * fieldSize + x) * 4;
      const nutrientValue = nutrientField[y][x] || 0;
      const toxinValue = toxinField[y][x] || 0;

      // Scale nutrient and toxin values to be between 0 and 255
      const nutrientIntensity = Math.min(255, nutrientValue * 255);
      const toxinIntensity = Math.min(255, toxinValue * 255);

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

function sendCommand(command) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ command: command }));
  } else {
    console.error("WebSocket is not open. Cannot send command:", command);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera); // Remove the if condition
}

// Function to update nutrient and toxin fields
function updateFields(nutrientField, toxinField) {
  const fieldSize = 200;
  const fieldMesh = scene.getObjectByName("fieldMesh");
  if (!fieldMesh) {
    console.error("fieldMesh not found");
    return;
  }

  // Create a new canvas texture if it doesn't exist
  if (!fieldMesh.material.map) {
    const fieldCanvas = document.createElement("canvas");
    fieldCanvas.width = fieldSize;
    fieldCanvas.height = fieldSize;
    const fieldTexture = new THREE.CanvasTexture(fieldCanvas);
    fieldMesh.material.map = fieldTexture;
  }

  const fieldTexture = fieldMesh.material.map;
  const fieldCanvas = fieldTexture.image;
  const fieldContext = fieldCanvas.getContext("2d");
  const imageData = fieldContext.createImageData(fieldSize, fieldSize);

  for (let x = 0; x < fieldSize; x++) {
    for (let y = 0; y < fieldSize; y++) {
      const index = (y * fieldSize + x) * 4;
      const nutrientValue = nutrientField[y][x] || 0;
      const toxinValue = toxinField[y][x] || 0;

      // Scale nutrient and toxin values to be between 0 and 255
      const nutrientIntensity = Math.min(255, nutrientValue * 255);
      const toxinIntensity = Math.min(255, toxinValue * 255);

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
function setupUI() {
  const controlsDiv = document.getElementById("controls");

  const startButton = document.createElement("button");
  startButton.innerText = "Start Simulation";
  startButton.onclick = function () {
    console.log("Start button clicked");
    simulationRunning = true;
    sendCommand("start");
  };
  controlsDiv.appendChild(startButton);

  const pauseButton = document.createElement("button");
  pauseButton.innerText = "Pause Simulation";
  pauseButton.onclick = function () {
    console.log("Pause button clicked");
    simulationRunning = false;
    sendCommand("pause");
  };
  controlsDiv.appendChild(pauseButton);

  const restartButton = document.createElement("button");
  restartButton.innerText = "Restart Simulation";
  restartButton.onclick = function () {
    console.log("Restart button clicked");
    simulationRunning = true;
    sendCommand("restart");
  };
  controlsDiv.appendChild(restartButton);
}
