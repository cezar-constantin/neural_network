# Neural network

Simulator interactiv pentru recunoasterea cifrelor scrise de mana si pentru vizualizarea activarii unei retele neuronale inspirate de prezentarea 3Blue1Brown "But what is a neural network?".

## Ce face

- ofera un canvas de desen pentru cifre scrise de mana;
- normalizeaza desenul la 28x28 pixeli, in stil MNIST;
- ruleaza inferenta pe o retea antrenata cu arhitectura `784 -> 16 -> 16 -> 10`;
- afiseaza activarea neuronilor pe fiecare layer;
- expune feature-map-uri pentru primul layer si proiectii compuse pentru al doilea layer.

## Arhitectura modelului

- input: 784 neuroni (`28 x 28`);
- hidden layer 1: 16 neuroni pentru stroke-uri si pattern-uri locale;
- hidden layer 2: 16 neuroni pentru parti de cifra, cum ar fi arce, bare si bucle;
- output: 10 neuroni pentru cifrele `0-9`.

Modelul din `model/model.json` a fost antrenat local pe exemple MNIST cu scriptul din `scripts/train-mnist.mjs`.

## Rulare locala

Proiectul este static, deci nu are nevoie de build.

1. Porneste un server local din radacina proiectului:

```powershell
python -m http.server 4173
```

2. Deschide:

```text
http://localhost:4173
```

## Reantrenare model

Datele brute MNIST nu sunt urcate in repo. Daca vrei sa refaci modelul:

1. descarca fisierele MNIST in folderul `data/`;
2. ruleaza:

```powershell
node scripts/train-mnist.mjs
```

Scriptul va genera `model/model.json`.

## Publicare pe GitHub Pages

Repo-ul include workflow-ul `.github/workflows/deploy-pages.yml`.

1. creeaza un repository nou pe GitHub;
2. impinge codul in branch-ul principal;
3. in GitHub, activeaza `Pages` cu sursa `GitHub Actions`;
4. dupa primul workflow finalizat, aplicatia va fi disponibila online.

## Fisiere importante

- `index.html` - structura interfetei;
- `styles.css` - identitatea vizuala si layout-ul;
- `app.js` - desen, preprocesare, inferenta si animatie;
- `model/model.json` - modelul antrenat;
- `model/examples.json` - exemple MNIST pentru demo;
- `scripts/train-mnist.mjs` - scriptul de antrenare.
