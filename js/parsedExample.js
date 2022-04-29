
sigmoid = (x) => { return 1 / (1 + Math.exp(-x)) };
_sigmoid = (x) => { return sigmoid(x) * (1 - sigmoid(x)) };


function MedNode() {
    this.bias = Math.random();
    this.weights = new Array();
    this.output = 0.0;
    this.sum = 0.0;
    this.dweights = new Array();
    this.delta = 0.0;
}


function MedNet (dataset = [{ inputs: [0, 0], outputs: [0] }], structure = [3], learning_rate = 0.15) {

    this.layers = new Array();
    this.learning_rate = learning_rate;
    this.inputs = new Array();
    this.outputs = new Array();
    this.targets = new Array();
    this.sigmoid = (x) => { return 1 / (1 + Math.exp(-x)) };
    this._sigmoid = (x) => { return this.sigmoid(x) * (1 - this.sigmoid(x)) };
    this.error = 0.0;

    dataset.forEach((item, index) => {
        this.inputs[index] = item.inputs;
        this.targets[index] = item.outputs;
    });
    //agrego los inputs al inicio de la estructura
    structure.unshift(dataset[0].inputs.length);

    //agrego los outputs al final de la estructura
    structure.push(dataset[0].outputs.length);

    var i, j, k;

    //inputs
    for (i = 0; i < structure.length; i++) {

        this.layers[i] = new Array();

        for (j = 0; j < structure[i]; j++) {
            this.layers[i][j] = new MedNode();
            if (i > 0) {
                for (k = 0; k < structure[i - 1]; k++) {
                    this.layers[i][j].weights[k] = Math.random();
                    this.layers[i][j].dweights[k] = 0.0;
                }
            }
            if (i == 0) {
                for (k = 0; k < structure[0]; k++) {
                    this.layers[i][j].weights[k] = Math.random();
                    this.layers[i][j].dweights[k] = 0.0;
                }
            }
        }
    }

    this.forward = (inputs = [0, 0]) => {

        let i, j, k;

        this.resetTempsVars();

        for (i = 0; i < this.layers[0].length; i++) {
            for (j = 0; j < inputs.length; j++) {
                this.layers[0][i].sum += inputs[j] * this.layers[0][i].weights[j];
            }
            this.layers[0][i].sum += this.layers[0][i].bias;
            this.layers[0][i].output = this.sigmoid(this.layers[0][i].sum);
        }
        

        for (i = 1; i < this.layers.length; i++) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = 0; k < this.layers[i - 1].length; k++) {
                    this.layers[i][j].sum += this.layers[i -1][k].output * this.layers[i][j].weights[k];
                }
                this.layers[i][j].sum += this.layers[i][j].bias;
                this.layers[i][j].output = this.sigmoid(this.layers[i][j].sum);
            }
        }



    }

    this.resetTempsVars = () => {
        let i, j, k;
        for (i = 0; i < this.layers.length; i++) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = 0; k < this.layers[i][j].dweights.length; k++) {
                    this.layers[i][j].dweights[k] = 0.0;
                }
                this.layers[i][j].sum = 0.0;
            }
        }
    }

    this.calculateError = () => {
        let i;
        let acu = 0.0;
        let lastLayer = this.layers.length - 1;

        for (i = 0; i < this.layers[lastLayer].length; i++) {
            acu += Math.pow((this.targets[i] - this.layers[lastLayer][i].output), 2);
        }
        this.error = acu / this.layers[0].length;
    }

    this.backward = (targets = [0]) => {
        let i, j, k, l, m;
        let lastLayer = this.layers.length - 1;

        for (i = 0; i < this.layers[lastLayer].length; i++) { 
            let delta = this.layers[lastLayer][i].output - targets[i];
            let output_delta = delta * this._sigmoid(this.layers[lastLayer][i].sum);
            this.layers[lastLayer][i].delta = output_delta;       
        }

        //recorremos en oden de capas para recalcular los pesos
        for (i = lastLayer - 1; i >= 0; i--) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = lastLayer; k >= 0; k--) {
                    for (l = 0; l < this.layers[k].length; l++) {
                        for (m = 0; m < this.layers[k][l].dweights.length; m++) {
                            this.layers[k][l].dweights[m] += this.layers[i][j].sum * this.layers[k][l].delta;
                            this.layers[k][l].bias += this.layers[k][l].delta;
                            this.layers[i][j].delta = this.layers[k][l].delta * this._sigmoid(this.layers[i][j].sum);
                        }
                    }
                }
            }
        }
    }

    this.update = () => {
        let i, j, k;

        for (i = 0; i < this.layers.length; i++) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = 0; k < this.layers[i][j].dweights.length; k++) {
                    let pretmp = parseFloat(this.learning_rate * this.layers[i][j].dweights[k]);
                    let tmp = parseFloat(this.layers[i][j].weights[k] - pretmp);
                    this.layers[i][j].weights[k] = tmp;
                }
                let dbias = this.layers[i][j].bias;
                this.layers[i][j].bias = (dbias - (this.learning_rate * dbias));
            }
        }
    }

    this.setOutputs = () => {
        let i;
        let lastLayer = this.layers.length - 1;

        for (i = 0; i < this.layers[lastLayer].length; i++) {
            this.outputs[i] = this.layers[lastLayer][i].output;
        }
    }

    this.train = (times = 1) => {
        for (var i = 0; i < times; i++) {
            this.inputs.forEach((element, index) => {
                this.forward(element);
                this.calculateError();
                this.setOutputs();
                console.table({inputs: element, targets: this.targets[index], outputs: this.outputs, error: this.error});

                //this.output();
                this.backward(this.targets[index]);
                this.update();
            });
        }
    }

    this.output = () => {

        this.setOutputs();
        this.calculateError();

        var output = {}
        output.inputs = this.inputs;
        output.targets = this.targets;
        output.outputs = this.outputs;
        output.error = this.error;

        console.table(output);
    }

    this.run = () => {
        this.forward();
        this.output();
    }
}


var net = new MedNet([
    { inputs: [0, 0], outputs: [1] },
    // { inputs: [1, 0], outputs: [1] },
    // { inputs: [0, 1], outputs: [1] },
    // { inputs: [1, 1], outputs: [0] },
], [3], 0.15);

net.train(500);
//net.run();

//console.log(JSON.stringify(net));


/*



var entrada1 = new Node();
var entrada2 = new Node();
var oculta1 = new Node();
var oculta2 = new Node();
var oculta3 = new Node();
var salida1 = new Node();

var error = 0.0;

var dataset = [
    {input: [1, 1], output: [1]},
    {input: [0, 1], output: [0]},
    {input: [1, 0], output: [0]},
    {input: [1, 1], output: [1]},
];



entrada1.weights[0] = Math.random();
entrada2.weights[0] = Math.random();

oculta1.weights[0] = Math.random();
oculta1.weights[1] = Math.random();

oculta2.weights[0] = Math.random();
oculta2.weights[1] = Math.random();

oculta3.weights[0] = Math.random();
oculta3.weights[1] = Math.random();

salida1.weights[0] = Math.random();
salida1.weights[1] = Math.random();
salida1.weights[2] = Math.random();




for(let i = 0; i < 2000; i++) {

    entrada1.sum = entrada2.sum = oculta1.sum = oculta2.sum = oculta3.sum = salida1.sum = 0.0;
    entrada1.dweights[0] = 0.0;
    entrada2.dweights[0] = 0.0;
    oculta1.dweights[0] = 0.0;
    oculta1.dweights[1] = 0.0;
    oculta2.dweights[0] = 0.0;
    oculta2.dweights[1] = 0.0;
    oculta3.dweights[0] = 0.0;
    oculta3.dweights[1] = 0.0;
    salida1.dweights[0] = 0.0;
    salida1.dweights[1] = 0.0;
    salida1.dweights[2] = 0.0;


    entrada1.sum = dataset[0].input[1] * entrada1.weights[0];
    entrada2.sum = dataset[0].input[0] * entrada2.weights[0];
    
    entrada1.output = sigmoid(entrada1.sum + entrada1.bias);
    entrada2.output = sigmoid(entrada2.sum + entrada2.bias);


    oculta1.sum += entrada1.output * oculta1.weights[0];
    oculta1.sum += entrada2.output * oculta1.weights[1];

    oculta1.output = sigmoid(oculta1.sum + oculta1.bias);

    oculta2.sum += entrada1.output * oculta2.weights[0];
    oculta2.sum += entrada2.output * oculta2.weights[1];

    oculta2.output = sigmoid(oculta2.sum + oculta2.bias);

    oculta3.sum += entrada1.output * oculta3.weights[0];
    oculta3.sum += entrada2.output * oculta3.weights[1];

    oculta3.output = sigmoid(oculta3.sum + oculta3.bias);

    salida1.sum += oculta1.output * salida1.weights[0];
    salida1.sum += oculta2.output * salida1.weights[1];
    salida1.sum += oculta3.output * salida1.weights[2];

    salida1.output = sigmoid(salida1.sum + salida1.bias);

    let delta = dataset[0].output[0] - salida1.output;

    salida1.delta = delta * _sigmoid(salida1.sum);

    oculta1.delta = salida1.delta * _sigmoid(oculta1.sum);
    oculta2.delta = salida1.delta * _sigmoid(oculta2.sum);
    oculta3.delta = salida1.delta * _sigmoid(oculta3.sum);

    entrada1.delta = salida1.delta * _sigmoid(entrada1.sum);
    entrada2.delta = salida1.delta * _sigmoid(entrada2.sum);

    salida1.dweights[0] += oculta1.sum * salida1.delta;
    salida1.dweights[1] += oculta2.sum * salida1.delta;
    salida1.dweights[2] += oculta3.sum * salida1.delta;
    salida1.bias += salida1.delta;

    oculta1.dweights[0] += entrada1.sum * oculta1.delta;
    oculta1.dweights[1] += entrada2.sum * oculta1.delta;
    oculta1.bias += oculta1.delta;

    oculta2.dweights[0] += entrada1.sum * oculta2.delta;
    oculta2.dweights[1] += entrada2.sum * oculta2.delta;
    oculta2.bias += oculta2.delta;

    oculta3.dweights[0] += entrada1.sum * oculta3.delta;
    oculta3.dweights[1] += entrada2.sum * oculta3.delta;
    oculta3.bias += oculta3.delta;

    entrada1.dweights[0] += entrada1.sum * entrada1.delta;
    entrada1.bias += entrada1.delta;

    entrada2.dweights[0] += entrada1.sum * entrada2.delta;
    entrada2.bias += entrada2.delta;


    salida1.weights[0] +=  salida1.dweights[0];
    salida1.weights[1] +=  salida1.dweights[1];
    salida1.weights[2] +=  salida1.dweights[2];

    oculta1.weights[0] +=  oculta1.dweights[0];
    oculta1.weights[1] +=  oculta1.dweights[1];

    oculta2.weights[0] +=  oculta2.dweights[0];
    oculta2.weights[1] +=  oculta2.dweights[1];

    oculta3.weights[0] +=  oculta3.dweights[0];
    oculta3.weights[1] +=  oculta3.dweights[1];

    entrada1.weights[0] +=  entrada1.dweights[0];
    entrada2.weights[0] +=  entrada2.dweights[0];

    error = (delta * delta) / 2;
    console.table({inputs: dataset[0].input, targets: dataset[0].output, output: salida1.output, error: error});
}

*/
