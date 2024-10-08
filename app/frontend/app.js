console.log("app.js is loaded");

let scene, camera, renderer;
let simulationRunning = false;
let socket;

document.addEventListener("DOMContentLoaded", (event) => {
  console.log("DOM fully loaded and parsed");
  initThreeJS();
  setupUI();
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
  document.body.appendChild(renderer.domElement);

  // Create a plane to represent the medium
  const tempGeometry = new THREE.PlaneGeometry(200, 200);
  const tempMaterial = new THREE.MeshBasicMaterial({
    color: 0xaaaaaa,
    side: THREE.DoubleSide,
  });
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
  const fieldMaterial = new THREE.MeshBasicMaterial({
    map: fieldTexture,
    transparent: true,
  });
  const fieldMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    fieldMaterial
  );
  scene.add(fieldMesh);

  camera.position.z = 300;

  animate();
}

function connectWebSocket() {
  console.log("Attempting to connect WebSocket");
  socket = new WebSocket("ws://localhost:8000/ws/simulation");

  socket.onopen = function (event) {
    console.log("WebSocket connection established");
  };

  socket.onmessage = function (event) {
    console.log("Received message:", event.data);
    // We'll handle the data here later
  };

  socket.onerror = function (error) {
    console.error("WebSocket error:", error);
  };

  socket.onclose = function (event) {
    console.log("WebSocket connection closed:", event);
  };
}

function animate() {
  requestAnimationFrame(animate);
  if (simulationRunning) {
    renderer.render(scene, camera);
  }
}

// Function to update nutrient and toxin fields
function updateFields(nutrientField, toxinField) {
  const fieldSize = 200;
  const fieldContext = scene
    .getObjectByName("fieldMesh")
    .material.map.image.getContext("2d");
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
  scene.getObjectByName("fieldMesh").material.map.needsUpdate = true;
}

// Handle window resizing
window.addEventListener("resize", function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// UI Controls for Simulation
function setupUI() {
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
  };
  controlsDiv.appendChild(restartButton);
}
