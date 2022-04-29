
sigmoid = (x) => { return 1 / (1 + Math.exp(-x)) };
_sigmoid = (x) => { return sigmoid(x) * (1 - sigmoid(x)) };


function Node() {
    this.bias = Math.random();
    this.weights = new Array();
    this.output = 0.0;
    this.sum = 0.0;
    this.dweights = new Array();
    this.delta = 0.0;
}

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