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
          }
        } else {
          for (k = 0; k < structure[i - 1]; k++) {
            this.layers[i][j].weights[k] = weightInitFn();
            this.layers[i][j].dweights[k] = weightInitFn();
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

  forward() {
    for (let i = 0; i < this.inputs.length; i++) {
      for (let j = 0; j < this.layers[0].length; j++) {
        this.layers[0][j].sum = this.layers[0][j].bias;
        for (let k = 0; k < this.layers[0][j].weights.length; k++) {
          this.layers[0][j].sum +=
            this.inputs[i][k] * this.layers[0][j].weights[k];
          console.log(
            `inputs[${i}][${k}]: ${this.inputs[i][k]} | layer[0][${j}]: ${this.layers[0][j].sum}`
          );
        }
        this.layers[0][j].output = this.activationFn(this.layers[0][j].sum);
      }

      for (let j = 1; j < this.layers.length; j++) {
        for (let k = 0; k < this.layers[j].length; k++) {
          this.layers[j][k].sum = this.layers[j][k].bias;
          for (let l = 0; l < this.layers[j - 1].length; l++) {
            this.layers[j][k].sum +=
              this.layers[j - 1][l].output * this.layers[j][k].weights[l];
          }
          this.layers[j][k].output = this.activationFn(this.layers[j][k].sum);
        }
      }
    }
  }

  backward() {
    var i, j, k, l, m, context, lc;
    context = this;
    lc = context.layers[context.layers.length - 1];

    this.targets.forEach((item, r) => {
      for (i = 0; i < lc.length; i++) {
        let delta = lc[i].output - item[i];
        lc[i].dsum = context.derivativeFn(lc[i].sum) * delta;
      }

      for (i = context.layers.length - 2; i >= 0; i--) {
        for (j = 0; j < context.layers[i].length; j++) {
          context.layers[i][j].doutput = 0;
          for (k = 0; k < context.layers[i + 1].length; k++) {
            for (l = 0; l < context.layers[i + 1][k].weights.length; l++) {
              context.layers[i][j].doutput +=
                context.layers[i + 1][k].dsum *
                context.layers[i + 1][k].weights[l];
            }
          }
          context.layers[i][j].dsum =
            context.derivativeFn(context.layers[i][j].sum) *
            context.layers[i][j].doutput;
        }
      }

      for (i = 1; i < context.layers.length; i++) {
        for (j = 0; j < context.layers[i].length; j++) {
          for (k = 0; k < context.layers[i][j].weights.length; k++) {
            context.layers[i][j].dweights[k] =
              context.layers[i][j].dweights[k] +
              context.layers[i - 1][k].output * context.layers[i][j].dsum;
          }
          context.layers[i][j].dbias =
            context.layers[i][j].dbias + context.layers[i][j].dsum;
        }
      }
    });

    for (i = 1; i < context.layers.length; i++) {
      for (j = 0; j < context.layers[i].length; j++) {
        for (k = 0; k < context.layers[i][j].weights.length; k++) {
          context.layers[i][j].weights[k] =
            context.layers[i][j].weights[k] -
            context.learning_rate * context.layers[i][j].dweights[k];
          context.layers[i][j].dweights[k] = 0;
        }
        context.layers[i][j].bias =
          context.layers[i][j].bias -
          context.learning_rate * context.layers[i][j].dbias;
        context.layers[i][j].dbias = 0;
      }
    }
  }

  update() {
    var i, j, k;

    for (i = 0; i < this.layers.length; i++) {
      for (j = 0; j < this.layers[i].length; j++) {
        for (k = 0; k < this.layers[i][j].weights.length; k++) {
          this.layers[i][j].weights[k] =
            this.layers[i][j].weights[k] -
            this.learning_rate * this.layers[i][j].dweights[k];
          this.layers[i][j].dweights[k] = 0;
        }
        this.layers[i][j].bias =
          this.layers[i][j].bias - this.learning_rate * this.layers[i][j].dbias;
        this.layers[i][j].doutput = 0;
        this.layers[i][j].dsum = 0;
      }
    }
  }

  train(iterations = 1000) {
    for (var o = 0; o < iterations; o++) {
      var lastLayer = this.layers[this.layers.length - 1];
      var i, j, k, l, m;

      for (i = 0; i < this.inputs.length; i++) {

        for (j = 0; j < this.layers[0].length; j++) {
          this.layers[0][j].sum = this.layers[0][j].bias;
          for (k = 0; k < this.layers[0][j].weights.length; k++) {
            this.layers[0][j].sum +=
              this.inputs[i][k] * this.layers[0][j].weights[k];
            // console.log(`inputs[${i}][${k}]: ${this.inputs[i][k]} | sum layer[0] neuron[${j}]: ${ this.layers[0][j].sum}`);
          }
          this.layers[0][j].output = this.activationFn(this.layers[0][j].sum);
        }

        for (j = 1; j < this.layers.length; j++) {
          for (k = 0; k < this.layers[j].length; k++) {
            this.layers[j][k].sum = this.layers[j][k].bias;
            for (l = 0; l < this.layers[j - 1].length; l++) {
              this.layers[j][k].sum +=
                this.layers[j - 1][l].output * this.layers[j][k].weights[l];
            }
            this.layers[j][k].output = this.activationFn(this.layers[j][k].sum);
          }
        }

        for (j = 0; j < lastLayer.length; j++) {
          let delta = lastLayer[j].output - this.targets[i][j];
          lastLayer[j].dsum = this.derivativeFn(lastLayer[j].sum) * delta;
        }

        for (j = this.layers.length - 2; j >= 0; j--) {
          for (m = 0; m < this.layers[j].length; m++) {
            this.layers[j][m].doutput = 0;
            for (k = 0; k < this.layers[j + 1].length; k++) {
              for (l = 0; l < this.layers[j + 1][k].weights.length; l++) {
                this.layers[j][m].doutput +=
                  this.layers[j + 1][k].dsum * this.layers[j + 1][k].weights[l];
              }
            }
            this.layers[j][m].dsum =
              this.derivativeFn(this.layers[j][m].sum) *
              this.layers[j][m].doutput;
          }
        }

        for (l = 1; l < this.layers.length; l++) {
          for (j = 0; j < this.layers[l].length; j++) {
            for (k = 0; k < this.layers[l][j].weights.length; k++) {
              this.layers[l][j].dweights[k] =
                this.layers[l][j].dweights[k] +
                this.layers[l - 1][k].output * this.layers[l][j].dsum;
            }
            this.layers[l][j].dbias =
              this.layers[l][j].dbias + this.layers[l][j].dsum;
          }
        }

        for (l = 1; l < this.layers.length; l++) {
          for (j = 0; j < this.layers[l].length; j++) {
            for (k = 0; k < this.layers[l][j].weights.length; k++) {
              this.layers[l][j].weights[k] =
                this.layers[l][j].weights[k] -
                this.learning_rate * this.layers[l][j].dweights[k];
              this.layers[l][j].dweights[k] = 0;
            }
            this.layers[l][j].bias =
              this.layers[l][j].bias -
              this.learning_rate * this.layers[l][j].dbias;
            this.layers[l][j].dbias = 0;
          }
        }

        // for (l = 0; l < this.layers.length; l++) {
        //   for (j = 0; j < this.layers[l].length; j++) {
        //     for (k = 0; k < this.layers[l][j].weights.length; k++) {
        //       this.layers[l][j].weights[k] =
        //         this.layers[l][j].weights[k] -
        //         this.learning_rate * this.layers[l][j].dweights[k];
        //       this.layers[l][j].dweights[k] = 0;
        //     }
        //     this.layers[l][j].bias =
        //       this.layers[l][j].bias -
        //       this.learning_rate * this.layers[l][j].dbias;
        //     this.layers[l][j].doutput = 0;
        //     this.layers[l][j].dsum = 0;
        //   }
        // }
      }
    }
  }

  output(inputs) {
    if (inputs.length !== this.layers[0].length) {
      throw new Error(
        "El número de entradas debe coincidir con el número de nodos en la capa de entrada"
      );
    }

    var i, j, k, sum;

    for (i = 0; i < inputs.length; i++) {
      this.layers[0][i].output = inputs[i];
    }

    for (i = 2; i < this.layers.length; i++) {
      for (j = 0; j < this.layers[i].length; j++) {
        sum = this.layers[i][j].bias;
        for (k = 0; k < this.layers[i - 1].length; k++) {
          sum += this.layers[i - 1][k].output * this.layers[i][j].weights[k];
        }
        this.layers[i][j].sum = sum;
        this.layers[i][j].output = this.activationFn(sum);
      }
    }

    var outputLayer = this.layers[this.layers.length - 1];
    var output = new Array(outputLayer.length);
    for (i = 0; i < outputLayer.length; i++) {
      output[i] = outputLayer[i].output;
    }
    return output;
  }
}

function iniciar() {
  // Crear la instancia de la red neuronal
  const net = new MedNet({
    structure: [2, 3, 1], // 2 nodos de entrada, 2 nodos ocultos y 1 nodo de salida
    learning_rate: 0.1, // Tasa de aprendizaje
  });

  // Definir el dataset de entrenamiento
  const trainingData = [
    // { inputs: [0, 0], outputs: [0] },
    { inputs: [0, 0], outputs: [0] },
    { inputs: [1, 1], outputs: [1] },
    // { inputs: [1, 1], outputs: [1] },
  ];

  // Entrenar la red neuronal
  net.setData(trainingData); // Establecer el dataset de entrenamiento

  net.train(1000000);

  // Evaluar la red neuronal
  //    console.log(net.output([0,0]));
  console.log(net.output([1, 1]));
  console.log(net.output([0, 0]));
  //    console.log(net.output([1,1]));
}
