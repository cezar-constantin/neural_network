# Neural network

Interactive simulator for handwritten digit recognition and neural network activation visualization, inspired by the 3Blue1Brown presentation "But what is a neural network?".

## What It Does

- provides a drawing canvas for handwritten digits;
- normalizes the drawing to 28x28 pixels in MNIST style;
- runs inference through a trained network with architecture `784 -> 16 -> 16 -> 10`;
- shows neuron activation across each layer;
- displays feature maps for the first layer and composite projections for the second layer.

## Model Architecture

- input: 784 neurons (`28 x 28`);
- hidden layer 1: 16 neurons for local strokes and pattern detection;
- hidden layer 2: 16 neurons for digit parts such as arcs, bars, and loops;
- output: 10 neurons for digits `0-9`.

The model in `model/model.json` was trained locally on MNIST examples using `scripts/train-mnist.mjs`.

## Local Run

The project is static, so it does not need a build step.

1. Start a local server from the project root:

```powershell
python -m http.server 4173
```

2. Open:

```text
http://localhost:4173
```

## Retraining The Model

The raw MNIST files are not committed to the repo. If you want to recreate the model:

1. download the MNIST files into the `data/` folder;
2. run:

```powershell
node scripts/train-mnist.mjs
```

The script will generate `model/model.json`.

## GitHub Pages Deployment

The repository includes the workflow `.github/workflows/deploy-pages.yml`.

1. create a new GitHub repository;
2. push the code to the main branch;
3. in GitHub, enable `Pages` with `GitHub Actions` as the source;
4. after the first successful workflow run, the app will be available online.

## Important Files

- `index.html` - UI structure;
- `styles.css` - visual identity and layout;
- `app.js` - drawing, preprocessing, inference, and animation;
- `model/model.json` - trained model;
- `model/examples.json` - MNIST examples for the demo;
- `scripts/train-mnist.mjs` - training script.
