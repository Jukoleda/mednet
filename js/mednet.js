window.onload = iniciar;

class MedNode {
  constructor() {
    this.bias = Math.random();
    this.weights = [];
    this.output = 0.0;
    this.sum = 0.0;
    this.dbias = 0.0;
    this.dweights = [];
    this.doutput = 0.0;
    this.dsum = 0.0;
  }
}

class MedNet {
  constructor({
    structure = [2, 3, 1],
    learning_rate = 0.05,
    activationFn = (x) => {
      return 1 / (1 + Math.exp(-x));
    },
    derivativeFn = (x) => {
      return activationFn(x) * (1 - activationFn(x));
    },
    weightInitFn = () => {
      return Math.random();
    },
  } = {}) {
    if (structure.length < 3) {
      throw new Error(
        "La estructura de la red neuronal debe tener al menos una capa oculta y una capa de salida"
      );
    }
    if (structure.some((x) => x <= 0)) {
      throw new Error(
        "El número de nodos en cada capa de la red neuronal debe ser mayor que cero"
      );
    }

    this.layers = new Array();
    this.learning_rate = learning_rate;
    this.activationFn = activationFn;
    this.derivativeFn = derivativeFn;
    this.weightInitFn = weightInitFn;
    this.structure = structure;

    var i, j, k;

    for (i = 0; i < structure.length; i++) {
      this.layers[i] = new Array();

      for (j = 0; j < structure[i]; j++) {
        this.layers[i][j] = new MedNode();
        this.layers[i][j].weights = new Array();
        this.layers[i][j].dweights = new Array();
        if (i == 0) {
          for (k = 0; k < structure[0]; k++) {
            this.layers[i][j].weights[k] = weightInitFn();
            this.layers[i][j].dweights[k] = weightInitFn();
            // this.layers[i][j].bias = weightInitFn();
          }
        } else {
          for (k = 0; k < structure[i - 1]; k++) {
            this.layers[i][j].weights[k] = weightInitFn();
            this.layers[i][j].dweights[k] = weightInitFn();
            // this.layers[i][j].bias = weightInitFn();
          }
        }
      }
    }
  }

  setData(dataset = [{ inputs: [0, 0], outputs: [0] }]) {
    this.inputs = new Array();
    this.targets = new Array();

    dataset.forEach((item, index) => {
      this.inputs[index] = item.inputs;
      this.targets[index] = item.outputs;
    });
  }

  sigmoid(x) {
    return this.activationFn(x);
  }

  _sigmoid(x) {
    return this.derivativeFn(x);
  }

  forward(inputs = [0, 0]) {
    for (let j = 0; j < this.layers[0].length; j++) {
      this.layers[0][j].sum = this.layers[0][j].bias;
      for (let i = 0; i < inputs.length; i++) {
        this.layers[0][j].sum += inputs[i] * this.layers[0][j].weights[i];
      }
      this.layers[0][j].output = this.activationFn(this.layers[0][j].sum);
    }

    for (let j = 1; j < this.layers.length; j++) {
      for (let k = 0; k < this.layers[j].length; k++) {
        this.layers[j][k].sum = this.layers[j][k].bias;
        for (let l = 0; l < this.layers[j - 1].length; l++) {
          this.layers[j][k].sum += this.layers[j - 1][l].output * this.layers[j][k].weights[l];
        }
        this.layers[j][k].output = this.activationFn(this.layers[j][k].sum);
      }
    }
    
  }

  backward(targets = [0]) {
    var i, j, k, l, m, context, lc;
    context = this;
    lc = context.layers[context.layers.length - 1];

    targets.forEach((item) => {
      for (i = 0; i < lc.length; i++) {
        let delta = lc[i].output - item;
        lc[i].dsum = context.derivativeFn(lc[i].sum) * delta;
      }

    });

      for (i = this.layers.length - 2; i >= 0; i--) {
        for (j = 0; j < this.layers[i].length; j++) {
          this.layers[i][j].doutput = 0;
          for (k = 0; k < this.layers[i + 1].length; k++) {
            for (l = 0; l < this.layers[i + 1][k].weights.length; l++) {
              this.layers[i][j].doutput += this.layers[i + 1][k].dsum * this.layers[i + 1][k].weights[l];
            }
          }
          this.layers[i][j].dsum = this.derivativeFn(this.layers[i][j].sum) * this.layers[i][j].doutput;
        }
      }

      for (i = 1; i < this.layers.length; i++) {
        for (j = 0; j < this.layers[i].length; j++) {
          for (k = 0; k < this.layers[i][j].weights.length; k++) {
            this.layers[i][j].dweights[k] =
              this.layers[i][j].dweights[k] +
              this.layers[i - 1][k].output * this.layers[i][j].dsum;
          }
          this.layers[i][j].dbias = this.layers[i][j].dbias + this.layers[i][j].dsum;
        }
      }
  

    for (i = 1; i < this.layers.length; i++) {
      for (j = 0; j < this.layers[i].length; j++) {
        for (k = 0; k < this.layers[i][j].weights.length; k++) {
          this.layers[i][j].weights[k] = this.layers[i][j].weights[k] - this.learning_rate * this.layers[i][j].dweights[k];
          this.layers[i][j].dweights[k] = 0;
        }
        this.layers[i][j].bias = this.layers[i][j].bias - this.learning_rate * this.layers[i][j].dbias;
      }
    }
  }



  train(iterations = 1000) {
    for (var o = 0; o < iterations; o++) {
      for(var i = 0; i < this.inputs.length; i++) {
        this.forward(this.inputs[i]);
        this.backward(this.targets[i]);
      }
    }
  }

  output(inputs) {
    if (inputs.length !== this.layers[0].length) {
      throw new Error(
        "El número de entradas debe coincidir con el número de nodos en la capa de entrada"
      );
    }

    this.forward(inputs);
    var outputLayer = this.layers[this.layers.length - 1];
    var output = new Array(outputLayer.length);
    for (var i = 0; i < outputLayer.length; i++) {
      output[i] = outputLayer[i].output;
    }
    return output;
  }
}

function iniciar() {
  // Crear la instancia de la red neuronal
  const net = new MedNet({
    structure: [3, 3, 1], // 2 nodos de entrada, 2 nodos ocultos y 1 nodo de salida
    learning_rate: 0.091, // Tasa de aprendizaje
  });

  // Definir el dataset de entrenamiento
  const trainingData = [
    { inputs: [0, 0, 0], outputs: [0] },
    { inputs: [0, 0, 1], outputs: [0] },
    { inputs: [0, 1, 1], outputs: [0] },
    { inputs: [1, 1, 1], outputs: [1] },
  ];

  // Entrenar la red neuronal
  net.setData(trainingData); // Establecer el dataset de entrenamiento

  net.train(1000000);

  // Evaluar la red neuronal
  console.log("[0, 0, 0] = 0: " + net.output([0, 0, 0]));
  console.log("[0, 0, 1] = 1: " + net.output([0, 0, 1]));
  console.log("[0, 1, 1] = 1: " + net.output([0, 1, 1]));
  console.log("[1, 1, 1] = 1: " + net.output([1, 1, 1]));

  console.log(JSON.stringify(net));
}
