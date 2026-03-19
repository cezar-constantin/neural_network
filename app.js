const SVG_NS = "http://www.w3.org/2000/svg";
const SOURCE_SIZE = 560;
const NORMALIZED_SIZE = 28;
const FEATURE_THUMBNAIL_SIZE = 84;
const INPUT_THRESHOLD = 12;

const COLORS = {
  ink: [23, 34, 43],
  muted: [120, 128, 136],
  bgDark: [8, 22, 29],
  warm: [255, 138, 72],
  sun: [242, 160, 61],
  teal: [29, 159, 138],
  cool: [87, 121, 255],
  output: [255, 107, 61],
  neutral: [211, 214, 218],
};

const state = {
  model: null,
  samples: [],
  drawing: false,
  lastPoint: null,
  predictionQueued: false,
  featureMapsLayer2: [],
  probabilityRows: [],
  featureCardsLayer1: [],
  featureCardsLayer2: [],
  network: {
    inputImage: null,
    hidden1Nodes: [],
    hidden2Nodes: [],
    outputNodes: [],
    inputEdges: [],
    hidden12Edges: [],
    hidden23Edges: [],
    maxAbsW2: 1,
    maxAbsW3: 1,
  },
};

const elements = {
  body: document.body,
  drawCanvas: document.getElementById("draw-canvas"),
  normalizedCanvas: document.getElementById("normalized-canvas"),
  clearButton: document.getElementById("clear-button"),
  sampleButton: document.getElementById("sample-button"),
  predictionDigit: document.getElementById("prediction-digit"),
  predictionLabel: document.getElementById("prediction-label"),
  predictionConfidence: document.getElementById("prediction-confidence"),
  probabilityBars: document.getElementById("probability-bars"),
  modelAccuracy: document.getElementById("model-accuracy"),
  networkSvg: document.getElementById("network-svg"),
  layer1Grid: document.getElementById("feature-grid-layer-1"),
  layer2Grid: document.getElementById("feature-grid-layer-2"),
  inputStatus: document.getElementById("input-status"),
  predictionPanel: document.querySelector(".prediction-panel"),
};

const drawContext = elements.drawCanvas.getContext("2d");
const normalizedContext = elements.normalizedCanvas.getContext("2d");
const scaleCanvas = document.createElement("canvas");
scaleCanvas.width = NORMALIZED_SIZE;
scaleCanvas.height = NORMALIZED_SIZE;
const scaleContext = scaleCanvas.getContext("2d");
const centeredCanvas = document.createElement("canvas");
centeredCanvas.width = NORMALIZED_SIZE;
centeredCanvas.height = NORMALIZED_SIZE;
const centeredContext = centeredCanvas.getContext("2d");
const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = NORMALIZED_SIZE;
sampleCanvas.height = NORMALIZED_SIZE;
const sampleContext = sampleCanvas.getContext("2d");

function createSvgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function relu(value) {
  return value > 0 ? value : 0;
}

function softmax(values) {
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
    }
  }

  const result = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const probability = Math.exp(values[i] - max);
    result[i] = probability;
    sum += probability;
  }

  for (let i = 0; i < result.length; i += 1) {
    result[i] /= sum;
  }

  return result;
}

function argmax(values) {
  let index = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[index]) {
      index = i;
    }
  }
  return index;
}

function mixColor(start, end, factor) {
  const amount = clamp(factor, 0, 1);
  return [
    Math.round(start[0] + (end[0] - start[0]) * amount),
    Math.round(start[1] + (end[1] - start[1]) * amount),
    Math.round(start[2] + (end[2] - start[2]) * amount),
  ];
}

function toRgba(color, alpha = 1) {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function toRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function normalizeLayer(values) {
  let max = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
    }
  }

  if (max <= 1e-9) {
    return values.map(() => 0);
  }

  return values.map((value) => Math.pow(value / max, 0.8));
}

function typedMatrix(matrix) {
  return matrix.map((row) => Float32Array.from(row));
}

function typedVector(vector) {
  return Float32Array.from(vector);
}

function prepareModel(payload) {
  return {
    ...payload,
    weights: {
      w1: typedMatrix(payload.weights.w1),
      b1: typedVector(payload.weights.b1),
      w2: typedMatrix(payload.weights.w2),
      b2: typedVector(payload.weights.b2),
      w3: typedMatrix(payload.weights.w3),
      b3: typedVector(payload.weights.b3),
    },
  };
}

function clearSourceCanvas() {
  drawContext.fillStyle = "black";
  drawContext.fillRect(0, 0, SOURCE_SIZE, SOURCE_SIZE);
}

function configureDrawContext() {
  drawContext.lineCap = "round";
  drawContext.lineJoin = "round";
  drawContext.strokeStyle = "rgba(255, 255, 255, 0.98)";
  drawContext.fillStyle = "rgba(255, 255, 255, 0.98)";
  drawContext.lineWidth = 34;
  drawContext.shadowBlur = 24;
  drawContext.shadowColor = "rgba(255, 255, 255, 0.28)";
}

function setupProbabilityBars() {
  for (let digit = 0; digit < 10; digit += 1) {
    const row = document.createElement("div");
    row.className = "probability-row";

    const label = document.createElement("span");
    label.textContent = digit;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);

    const value = document.createElement("strong");
    value.textContent = "0%";

    row.append(label, track, value);
    elements.probabilityBars.appendChild(row);
    state.probabilityRows.push({ fill, value });
  }
}

function createFeatureCard(container, label, layerClass) {
  const card = document.createElement("article");
  card.className = `feature-card ${layerClass}`;

  const header = document.createElement("div");
  header.className = "feature-card-header";

  const title = document.createElement("div");
  title.className = "feature-card-title";
  title.textContent = label;

  const value = document.createElement("div");
  value.className = "feature-card-value";
  value.textContent = "0% active";

  const canvas = document.createElement("canvas");
  canvas.width = FEATURE_THUMBNAIL_SIZE;
  canvas.height = FEATURE_THUMBNAIL_SIZE;

  header.append(title, value);
  card.append(header, canvas);
  container.appendChild(card);

  return { card, canvas, value };
}

function setupFeatureGrids() {
  for (let index = 0; index < 16; index += 1) {
    state.featureCardsLayer1.push(
      createFeatureCard(elements.layer1Grid, `Neuron ${String(index + 1).padStart(2, "0")}`, "is-layer-1"),
    );
    state.featureCardsLayer2.push(
      createFeatureCard(elements.layer2Grid, `Neuron ${String(index + 1).padStart(2, "0")}`, "is-layer-2"),
    );
  }
}

function createLayerTitle(container, x, title, subtitle) {
  const titleText = createSvgElement("text", {
    x,
    y: 86,
    class: "network-label",
    "text-anchor": "middle",
  });
  titleText.textContent = title;

  const subtitleText = createSvgElement("text", {
    x,
    y: 110,
    class: "network-subtitle",
    "text-anchor": "middle",
  });
  subtitleText.textContent = subtitle;

  container.append(titleText, subtitleText);
}

function createLayerCard(container, x, y, width, height) {
  const card = createSvgElement("rect", {
    x,
    y,
    width,
    height,
    rx: 28,
    class: "network-layer-card",
    fill: "rgba(255,255,255,0.55)",
    stroke: "rgba(23, 34, 43, 0.12)",
  });
  container.appendChild(card);
  return card;
}

function createNetworkNode(svg, x, y, radius, label, accent, valueOffset = 44) {
  const group = createSvgElement("g", {});
  const circle = createSvgElement("circle", {
    cx: x,
    cy: y,
    r: radius,
    class: "network-node-circle",
    fill: "rgba(211, 214, 218, 0.85)",
    stroke: toRgba(accent, 0.35),
    "stroke-width": 2,
  });

  const text = createSvgElement("text", {
    x,
    y: y + 1,
    class: "network-node-text",
  });
  text.textContent = label;

  const value = createSvgElement("text", {
    x,
    y: y + valueOffset,
    class: "network-node-value",
  });
  value.textContent = "0%";

  group.append(circle, text, value);
  svg.appendChild(group);
  return { group, circle, value, x, y, radius, accent };
}

function buildNetworkSvg() {
  const svg = elements.networkSvg;
  svg.innerHTML = "";

  const defs = createSvgElement("defs");
  const inputFilter = createSvgElement("filter", { id: "input-shadow", x: "-20%", y: "-20%", width: "140%", height: "140%" });
  inputFilter.appendChild(
    createSvgElement("feDropShadow", {
      dx: 0,
      dy: 18,
      stdDeviation: 18,
      "flood-color": "#1d9f8a",
      "flood-opacity": 0.18,
    }),
  );
  defs.appendChild(inputFilter);
  svg.appendChild(defs);

  const backgroundLayer = createSvgElement("g", {});
  const edgeLayer = createSvgElement("g", {});
  const nodeLayer = createSvgElement("g", {});
  const labelLayer = createSvgElement("g", {});
  svg.append(backgroundLayer, edgeLayer, nodeLayer, labelLayer);

  createLayerTitle(labelLayer, 150, "Input", "28x28 grayscale image");
  createLayerTitle(labelLayer, 390, "Hidden layer 1", "stroke and edge detectors");
  createLayerTitle(labelLayer, 690, "Hidden layer 2", "bars, loops, arcs and ovals");
  createLayerTitle(labelLayer, 950, "Output", "digits 0 to 9");

  createLayerCard(backgroundLayer, 42, 142, 210, 262);
  createLayerCard(backgroundLayer, 286, 142, 210, 262);
  createLayerCard(backgroundLayer, 586, 142, 210, 262);
  createLayerCard(backgroundLayer, 862, 120, 176, 444);

  const inputFrame = createSvgElement("rect", {
    x: 68,
    y: 178,
    width: 158,
    height: 158,
    rx: 20,
    class: "network-input-frame",
    fill: "#0b161d",
    stroke: "rgba(255,255,255,0.12)",
    filter: "url(#input-shadow)",
  });

  const inputImage = createSvgElement("image", {
    x: 79,
    y: 189,
    width: 136,
    height: 136,
    href: "",
    preserveAspectRatio: "none",
  });
  inputImage.style.imageRendering = "pixelated";

  const inputCaption = createSvgElement("text", {
    x: 147,
    y: 370,
    class: "network-subtitle",
    "text-anchor": "middle",
  });
  inputCaption.textContent = "normalized before inference";

  nodeLayer.append(inputFrame, inputImage, inputCaption);
  state.network.inputImage = inputImage;

  const hidden1Positions = [];
  const hidden2Positions = [];
  const hiddenStartX = 338;
  const hidden2StartX = 638;
  const hiddenStartY = 190;
  const gapX = 54;
  const gapY = 54;

  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      hidden1Positions.push({ x: hiddenStartX + col * gapX, y: hiddenStartY + row * gapY });
      hidden2Positions.push({ x: hidden2StartX + col * gapX, y: hiddenStartY + row * gapY });
    }
  }

  state.network.hidden1Nodes = hidden1Positions.map((position, index) =>
    createNetworkNode(nodeLayer, position.x, position.y, 19, index + 1, COLORS.sun),
  );
  state.network.hidden2Nodes = hidden2Positions.map((position, index) =>
    createNetworkNode(nodeLayer, position.x, position.y, 19, index + 1, COLORS.teal),
  );

  const outputStartY = 150;
  const outputGapY = 38;
  state.network.outputNodes = Array.from({ length: 10 }, (_, index) =>
    createNetworkNode(nodeLayer, 950, outputStartY + index * outputGapY, 19, index, COLORS.output, 36),
  );

  const inputAnchorX = 226;
  const inputAnchorY = 257;

  state.network.inputEdges = state.network.hidden1Nodes.map((node) => {
    const line = createSvgElement("path", {
      d: `M ${inputAnchorX} ${inputAnchorY} C 268 ${inputAnchorY}, 292 ${node.y}, ${node.x - node.radius} ${node.y}`,
      class: "network-edge",
      stroke: "rgba(242, 160, 61, 0.14)",
      "stroke-width": 2.4,
      opacity: 0.28,
    });
    edgeLayer.appendChild(line);
    return line;
  });

  state.network.hidden12Edges = [];
  for (let target = 0; target < state.network.hidden2Nodes.length; target += 1) {
    for (let source = 0; source < state.network.hidden1Nodes.length; source += 1) {
      const start = state.network.hidden1Nodes[source];
      const end = state.network.hidden2Nodes[target];
      const controlX = (start.x + end.x) / 2;
      const path = createSvgElement("path", {
        d: `M ${start.x + start.radius} ${start.y} C ${controlX} ${start.y}, ${controlX} ${end.y}, ${end.x - end.radius} ${end.y}`,
        class: "network-edge",
        stroke: "rgba(23, 34, 43, 0.07)",
        "stroke-width": 1.1,
        opacity: 0.3,
      });
      edgeLayer.appendChild(path);
      state.network.hidden12Edges.push({
        path,
        source,
        target,
        weight: state.model.weights.w2[target][source],
      });
    }
  }

  state.network.hidden23Edges = [];
  for (let target = 0; target < state.network.outputNodes.length; target += 1) {
    for (let source = 0; source < state.network.hidden2Nodes.length; source += 1) {
      const start = state.network.hidden2Nodes[source];
      const end = state.network.outputNodes[target];
      const controlX = (start.x + end.x) / 2;
      const path = createSvgElement("path", {
        d: `M ${start.x + start.radius} ${start.y} C ${controlX} ${start.y}, ${controlX} ${end.y}, ${end.x - end.radius} ${end.y}`,
        class: "network-edge",
        stroke: "rgba(23, 34, 43, 0.07)",
        "stroke-width": 1.1,
        opacity: 0.3,
      });
      edgeLayer.appendChild(path);
      state.network.hidden23Edges.push({
        path,
        source,
        target,
        weight: state.model.weights.w3[target][source],
      });
    }
  }

  state.network.maxAbsW2 = Math.max(
    ...state.network.hidden12Edges.map((edge) => Math.abs(edge.weight)),
    0.0001,
  );
  state.network.maxAbsW3 = Math.max(
    ...state.network.hidden23Edges.map((edge) => Math.abs(edge.weight)),
    0.0001,
  );
}

function drawHeatmap(canvas, values, accent) {
  const context = canvas.getContext("2d");
  const pixelCanvas = document.createElement("canvas");
  pixelCanvas.width = NORMALIZED_SIZE;
  pixelCanvas.height = NORMALIZED_SIZE;
  const pixelContext = pixelCanvas.getContext("2d");
  const imageData = pixelContext.createImageData(NORMALIZED_SIZE, NORMALIZED_SIZE);

  let maxAbs = 0;
  for (let i = 0; i < values.length; i += 1) {
    const absValue = Math.abs(values[i]);
    if (absValue > maxAbs) {
      maxAbs = absValue;
    }
  }
  maxAbs = Math.max(maxAbs, 1e-6);

  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    const amount = Math.pow(Math.abs(value) / maxAbs, 0.72);
    const color = mixColor(COLORS.bgDark, value >= 0 ? accent : COLORS.cool, amount);
    const offset = i * 4;
    imageData.data[offset] = color[0];
    imageData.data[offset + 1] = color[1];
    imageData.data[offset + 2] = color[2];
    imageData.data[offset + 3] = 255;
  }

  pixelContext.putImageData(imageData, 0, 0);
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(pixelCanvas, 0, 0, canvas.width, canvas.height);
}

function projectLayerTwoFeatures(model) {
  return model.weights.w2.map((row) => {
    const composite = new Float32Array(NORMALIZED_SIZE * NORMALIZED_SIZE);
    for (let source = 0; source < row.length; source += 1) {
      const weight = row[source];
      const sourceWeights = model.weights.w1[source];
      for (let pixel = 0; pixel < composite.length; pixel += 1) {
        composite[pixel] += sourceWeights[pixel] * weight;
      }
    }
    return composite;
  });
}

function renderFeatureMaps() {
  state.model.weights.w1.forEach((weights, index) => {
    drawHeatmap(state.featureCardsLayer1[index].canvas, weights, COLORS.warm);
  });
  state.featureMapsLayer2.forEach((weights, index) => {
    drawHeatmap(state.featureCardsLayer2[index].canvas, weights, COLORS.teal);
  });
}

function setFeatureActivity(cards, activations, accent) {
  const normalized = normalizeLayer(activations);
  cards.forEach((card, index) => {
    const intensity = normalized[index];
    card.card.style.setProperty("--activation", intensity.toFixed(3));
    card.card.style.borderColor = toRgba(accent, 0.1 + intensity * 0.4);
    card.value.textContent = `${Math.round(intensity * 100)}% active`;
  });
}

function setNodeVisual(node, intensity, accent, valueText) {
  const fill = mixColor(COLORS.neutral, accent, clamp(0.12 + intensity * 0.88, 0, 1));
  node.circle.setAttribute("fill", toRgb(fill));
  node.circle.setAttribute("stroke", toRgba(accent, 0.22 + intensity * 0.52));
  node.circle.setAttribute("r", (node.radius * (1 + intensity * 0.16)).toFixed(2));
  node.value.textContent = valueText;
}

function updateNetworkEdges(hidden1Norm, hidden2Norm, probabilities) {
  state.network.inputEdges.forEach((path, index) => {
    const intensity = hidden1Norm[index];
    path.setAttribute("stroke", toRgba(COLORS.sun, 0.12 + intensity * 0.48));
    path.setAttribute("stroke-width", (1.6 + intensity * 2.6).toFixed(2));
    path.setAttribute("opacity", (0.18 + intensity * 0.82).toFixed(3));
  });

  state.network.hidden12Edges.forEach((edge) => {
    const strength = Math.abs(edge.weight) / state.network.maxAbsW2;
    const intensity = hidden1Norm[edge.source] * hidden2Norm[edge.target] * Math.pow(strength, 0.75);
    const color = edge.weight >= 0 ? COLORS.warm : COLORS.cool;
    edge.path.setAttribute("stroke", toRgba(color, 0.04 + intensity * 0.88));
    edge.path.setAttribute("stroke-width", (0.7 + intensity * 2.2).toFixed(2));
    edge.path.setAttribute("opacity", (0.12 + intensity * 0.88).toFixed(3));
  });

  state.network.hidden23Edges.forEach((edge) => {
    const strength = Math.abs(edge.weight) / state.network.maxAbsW3;
    const intensity = hidden2Norm[edge.source] * probabilities[edge.target] * Math.pow(strength, 0.75);
    const color = edge.weight >= 0 ? COLORS.teal : COLORS.cool;
    edge.path.setAttribute("stroke", toRgba(color, 0.05 + intensity * 0.9));
    edge.path.setAttribute("stroke-width", (0.7 + intensity * 2.5).toFixed(2));
    edge.path.setAttribute("opacity", (0.12 + intensity * 0.88).toFixed(3));
  });
}

function setNetworkState(result) {
  if (!result) {
    state.network.hidden1Nodes.forEach((node) => setNodeVisual(node, 0, COLORS.sun, "0%"));
    state.network.hidden2Nodes.forEach((node) => setNodeVisual(node, 0, COLORS.teal, "0%"));
    state.network.outputNodes.forEach((node) => setNodeVisual(node, 0, COLORS.output, "0%"));
    updateNetworkEdges(new Array(16).fill(0), new Array(16).fill(0), new Array(10).fill(0));
    if (state.network.inputImage) {
      state.network.inputImage.setAttribute("href", "");
    }
    return;
  }

  const hidden1Norm = normalizeLayer(result.a1);
  const hidden2Norm = normalizeLayer(result.a2);

  state.network.hidden1Nodes.forEach((node, index) =>
    setNodeVisual(node, hidden1Norm[index], COLORS.sun, `${Math.round(hidden1Norm[index] * 100)}%`),
  );
  state.network.hidden2Nodes.forEach((node, index) =>
    setNodeVisual(node, hidden2Norm[index], COLORS.teal, `${Math.round(hidden2Norm[index] * 100)}%`),
  );
  state.network.outputNodes.forEach((node, index) =>
    setNodeVisual(node, result.probabilities[index], COLORS.output, `${Math.round(result.probabilities[index] * 100)}%`),
  );

  updateNetworkEdges(hidden1Norm, hidden2Norm, Array.from(result.probabilities));
  state.network.inputImage.setAttribute("href", centeredCanvas.toDataURL("image/png"));
}

function forwardPass(model, input) {
  const { w1, b1, w2, b2, w3, b3 } = model.weights;

  const z1 = new Float32Array(16);
  const a1 = new Float32Array(16);
  for (let neuron = 0; neuron < 16; neuron += 1) {
    let sum = b1[neuron];
    const weights = w1[neuron];
    for (let pixel = 0; pixel < input.length; pixel += 1) {
      sum += weights[pixel] * input[pixel];
    }
    z1[neuron] = sum;
    a1[neuron] = relu(sum);
  }

  const z2 = new Float32Array(16);
  const a2 = new Float32Array(16);
  for (let neuron = 0; neuron < 16; neuron += 1) {
    let sum = b2[neuron];
    const weights = w2[neuron];
    for (let source = 0; source < a1.length; source += 1) {
      sum += weights[source] * a1[source];
    }
    z2[neuron] = sum;
    a2[neuron] = relu(sum);
  }

  const z3 = new Float32Array(10);
  for (let neuron = 0; neuron < 10; neuron += 1) {
    let sum = b3[neuron];
    const weights = w3[neuron];
    for (let source = 0; source < a2.length; source += 1) {
      sum += weights[source] * a2[source];
    }
    z3[neuron] = sum;
  }

  const probabilities = softmax(z3);
  return { z1, a1, z2, a2, z3, probabilities };
}

function getPointerPosition(event) {
  const rect = elements.drawCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * SOURCE_SIZE;
  const y = ((event.clientY - rect.top) / rect.height) * SOURCE_SIZE;
  return {
    x: clamp(x, 0, SOURCE_SIZE),
    y: clamp(y, 0, SOURCE_SIZE),
  };
}

function drawDot(point) {
  drawContext.beginPath();
  drawContext.arc(point.x, point.y, drawContext.lineWidth * 0.3, 0, Math.PI * 2);
  drawContext.fill();
}

function drawSegment(from, to) {
  drawContext.beginPath();
  drawContext.moveTo(from.x, from.y);
  drawContext.lineTo(to.x, to.y);
  drawContext.stroke();
}

function queuePrediction() {
  if (state.predictionQueued) {
    return;
  }
  state.predictionQueued = true;
  requestAnimationFrame(() => {
    state.predictionQueued = false;
    runPrediction();
  });
}

function getBoundsFromImage(imageData) {
  const { data, width, height } = imageData;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const value = data[offset];
      if (value > INPUT_THRESHOLD) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX === -1) {
    return null;
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
  };
}

function centerNormalizedDigit() {
  const imageData = scaleContext.getImageData(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const { data } = imageData;
  let mass = 0;
  let centerX = 0;
  let centerY = 0;

  for (let y = 0; y < NORMALIZED_SIZE; y += 1) {
    for (let x = 0; x < NORMALIZED_SIZE; x += 1) {
      const offset = (y * NORMALIZED_SIZE + x) * 4;
      const intensity = data[offset] / 255;
      mass += intensity;
      centerX += x * intensity;
      centerY += y * intensity;
    }
  }

  centeredContext.fillStyle = "black";
  centeredContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);

  if (mass <= 1e-9) {
    return new Float32Array(NORMALIZED_SIZE * NORMALIZED_SIZE);
  }

  const shiftX = Math.round(13.5 - centerX / mass);
  const shiftY = Math.round(13.5 - centerY / mass);
  centeredContext.drawImage(scaleCanvas, shiftX, shiftY);

  const centeredImage = centeredContext.getImageData(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const input = new Float32Array(NORMALIZED_SIZE * NORMALIZED_SIZE);

  for (let index = 0; index < input.length; index += 1) {
    const intensity = centeredImage.data[index * 4] / 255;
    input[index] = Math.pow(intensity, 0.9);
  }

  normalizedContext.imageSmoothingEnabled = false;
  normalizedContext.clearRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  normalizedContext.drawImage(centeredCanvas, 0, 0);

  return input;
}

function preprocessDrawing() {
  const sourceImage = drawContext.getImageData(0, 0, SOURCE_SIZE, SOURCE_SIZE);
  const bounds = getBoundsFromImage(sourceImage);
  if (!bounds) {
    scaleContext.fillStyle = "black";
    scaleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    centeredContext.fillStyle = "black";
    centeredContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    normalizedContext.fillStyle = "black";
    normalizedContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    return { input: new Float32Array(NORMALIZED_SIZE * NORMALIZED_SIZE), empty: true };
  }

  const targetScale = 20 / Math.max(bounds.width, bounds.height);
  const targetWidth = Math.max(1, Math.round(bounds.width * targetScale));
  const targetHeight = Math.max(1, Math.round(bounds.height * targetScale));

  scaleContext.fillStyle = "black";
  scaleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  scaleContext.imageSmoothingEnabled = true;
  scaleContext.drawImage(
    elements.drawCanvas,
    bounds.minX,
    bounds.minY,
    bounds.width,
    bounds.height,
    (NORMALIZED_SIZE - targetWidth) / 2,
    (NORMALIZED_SIZE - targetHeight) / 2,
    targetWidth,
    targetHeight,
  );

  return {
    input: centerNormalizedDigit(),
    empty: false,
  };
}

function setProbabilityBars(probabilities) {
  probabilities.forEach((probability, index) => {
    const percent = Math.round(probability * 100);
    state.probabilityRows[index].fill.style.width = `${percent}%`;
    state.probabilityRows[index].value.textContent = `${percent}%`;
  });
}

function setPredictionEmpty() {
  elements.predictionPanel.classList.add("empty-state");
  elements.predictionDigit.textContent = "?";
  elements.predictionLabel.textContent = "Astept desenul tau";
  elements.predictionConfidence.textContent = "Confidenta: 0%";
  normalizedContext.fillStyle = "black";
  normalizedContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  setProbabilityBars(new Array(10).fill(0));
  setNetworkState(null);
  setFeatureActivity(state.featureCardsLayer1, new Array(16).fill(0), COLORS.warm);
  setFeatureActivity(state.featureCardsLayer2, new Array(16).fill(0), COLORS.teal);
  elements.inputStatus.textContent = "Canvas gol";
}

function setPredictionResult(result) {
  const predictedDigit = argmax(result.probabilities);
  const confidence = result.probabilities[predictedDigit];
  elements.predictionPanel.classList.remove("empty-state");
  elements.predictionDigit.textContent = predictedDigit;
  elements.predictionLabel.textContent = `Modelul vede cel mai probabil cifra ${predictedDigit}`;
  elements.predictionConfidence.textContent = `Confidenta: ${Math.round(confidence * 100)}%`;
  setProbabilityBars(Array.from(result.probabilities));
  setNetworkState(result);
  setFeatureActivity(state.featureCardsLayer1, Array.from(result.a1), COLORS.warm);
  setFeatureActivity(state.featureCardsLayer2, Array.from(result.a2), COLORS.teal);
  elements.inputStatus.textContent = "Predictie live";
}

function runPrediction() {
  if (!state.model) {
    return;
  }

  const processed = preprocessDrawing();
  if (processed.empty) {
    setPredictionEmpty();
    return;
  }

  const result = forwardPass(state.model, processed.input);
  setPredictionResult(result);
}

function startDrawing(event) {
  event.preventDefault();
  elements.drawCanvas.setPointerCapture(event.pointerId);
  state.drawing = true;
  const point = getPointerPosition(event);
  state.lastPoint = point;
  drawDot(point);
  queuePrediction();
}

function continueDrawing(event) {
  if (!state.drawing) {
    return;
  }
  event.preventDefault();
  const point = getPointerPosition(event);
  drawSegment(state.lastPoint, point);
  state.lastPoint = point;
  queuePrediction();
}

function stopDrawing(event) {
  if (!state.drawing) {
    return;
  }
  state.drawing = false;
  state.lastPoint = null;
  if (event && elements.drawCanvas.hasPointerCapture(event.pointerId)) {
    elements.drawCanvas.releasePointerCapture(event.pointerId);
  }
  queuePrediction();
}

function clearDrawing() {
  clearSourceCanvas();
  setPredictionEmpty();
}

function renderSample(sample) {
  sampleContext.fillStyle = "black";
  sampleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const sampleImage = sampleContext.createImageData(NORMALIZED_SIZE, NORMALIZED_SIZE);
  for (let i = 0; i < sample.pixels.length; i += 1) {
    const value = Math.round(sample.pixels[i] * 255);
    sampleImage.data[i * 4] = value;
    sampleImage.data[i * 4 + 1] = value;
    sampleImage.data[i * 4 + 2] = value;
    sampleImage.data[i * 4 + 3] = 255;
  }
  sampleContext.putImageData(sampleImage, 0, 0);

  clearSourceCanvas();
  drawContext.imageSmoothingEnabled = true;
  drawContext.drawImage(sampleCanvas, 84, 84, 392, 392);
  queuePrediction();
}

function useRandomSample() {
  if (!state.samples.length) {
    return;
  }
  const sample = state.samples[Math.floor(Math.random() * state.samples.length)];
  renderSample(sample);
}

async function loadSamples() {
  try {
    const response = await fetch("./model/examples.json");
    if (!response.ok) {
      throw new Error("Examples unavailable");
    }
    const samples = await response.json();
    state.samples = samples;
    elements.sampleButton.disabled = false;
  } catch (error) {
    elements.sampleButton.disabled = true;
    elements.sampleButton.textContent = "Exemple indisponibile";
  }
}

function attachCanvasEvents() {
  elements.drawCanvas.addEventListener("pointerdown", startDrawing);
  elements.drawCanvas.addEventListener("pointermove", continueDrawing);
  elements.drawCanvas.addEventListener("pointerup", stopDrawing);
  elements.drawCanvas.addEventListener("pointerleave", stopDrawing);
  elements.drawCanvas.addEventListener("pointercancel", stopDrawing);
  elements.clearButton.addEventListener("click", clearDrawing);
  elements.sampleButton.addEventListener("click", useRandomSample);
}

async function loadModel() {
  const response = await fetch("./model/model.json");
  if (!response.ok) {
    throw new Error("Nu am putut incarca modelul.");
  }
  const payload = await response.json();
  state.model = prepareModel(payload);
  state.featureMapsLayer2 = projectLayerTwoFeatures(state.model);
  elements.modelAccuracy.textContent = `${Math.round(payload.metrics.testAccuracy * 100)}% pe ${payload.metrics.testSamples} exemple`;
}

async function initialize() {
  setupProbabilityBars();
  setupFeatureGrids();
  configureDrawContext();
  clearSourceCanvas();

  try {
    await loadModel();
    buildNetworkSvg();
    renderFeatureMaps();
    await loadSamples();
    attachCanvasEvents();
    setPredictionEmpty();
  } catch (error) {
    elements.modelAccuracy.textContent = "Modelul nu s-a incarcat";
    elements.inputStatus.textContent = "Eroare la initializare";
    elements.predictionLabel.textContent = "Modelul lipseste";
    elements.predictionConfidence.textContent = "Verifica fisierele din folderul model/";
    console.error(error);
  }
}

initialize();
