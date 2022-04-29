//window.onload = iniciar;

// evals = [{inputs: [0,0], output: 0}]

//se multiplica delta * output
//dw = output * delta
//dbias  = delta
//var h1_delta = o1_delta * (output * (1 - output))



class MedNode {
    constructor() {
        this.bias = Math.random();
        this.weights = new Array();
        this.output = 0.0;
        this.sum = 0.0;
        this.dbias = 0.0;
        this.dweights = new Array();
        this.delta = 0.0; //delta =  output - o1 * o1 * (1 - o1);
    }
}

class MedNet {
    constructor(dataset = [{ inputs: [0, 0], outputs: [0] }], structure = [3], learning_rate = 0.15) {

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
                        this.layers[i][j].dweights[k] = 0.0;//Math.random();
                    }
                }
                if (i == 0) {
                    this.layers[i][j].weights[0] = Math.random();
                    this.layers[i][j].dweights[0] = 0.0;//Math.random();
                }
            }
        }

    }

    forward() {

        let i, j, k, r;
        let acu = 0.0;
        let lastLayer = this.layers.length - 1;

        for (i = 0; i < this.layers.length; i++) {
            for (j = 0; j < this.layers[i]; j++) {
                for (k = 0; k < this.layers[i][j].dweights.length; k++) {
                    this.layers[i][j].dweights[k] = 0.0;
                }
                this.layers[i][j].sum = 0.0;
            }
        }

        
        for (r = 0; r < this.inputs.length; r++) {
            
            for (i = 0; i < this.layers[0].length; i++) {
                this.layers[0][i].sum = parseFloat(this.layers[0][i].bias + (this.inputs[r][i] * this.layers[0][i].weights[0]));
                this.layers[0][i].output = this.sigmoid(this.layers[0][i].sum);
            }
            
            for (i = 1; i < this.layers.length; i++) {
                for (j = 0; j < this.layers[i].length; j++) {
                    for (k = 0; k < this.layers[i - 1].length; k++) {
                        this.layers[i][j].sum += (this.layers[i - 1][k].output * this.layers[i][j].weights[k]);
                    }
                    this.layers[i][j].sum += this.layers[i][j].bias;
                    this.layers[i][j].output = this.sigmoid(this.layers[i][j].sum);
                }
            }
            for (i = 0; i < this.layers[lastLayer].length; i++) {
                this.outputs[r] = this.layers[lastLayer][i].output;
                acu += Math.pow((this.targets[r] - this.layers[lastLayer][i].output), 2);
            }
        }

        this.error = acu / this.layers[0].length;
    }

    backward() {
        let i, j, k, l, m;
        let lastLayer = this.layers.length - 1;
        
        //calculamos el delta de la ultima capa para empezar el aprendizaje
        for (i = 0; i < this.targets.length; i++) {
            for (j = 0; j < this.layers[lastLayer].length; j++) {
                //target - output
                let delta = this.layers[lastLayer][j].output - this.targets[i];
                let output_delta = delta * this._sigmoid(this.layers[lastLayer][j].sum);
                this.layers[lastLayer][j].delta = output_delta;
            }
        }

        //recorremos en oden de capas para recalcular los pesos
        for (i = lastLayer - 1; i >= 0; i--) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = lastLayer; k >= 0; k--) {
                    for (l = 0; l < this.layers[k].length; l++) {
                        for (m = 0; m < this.layers[k][l].dweights.length; m++) {
                            this.layers[k][l].dweights[m] += this.layers[i][j].output * this.layers[k][l].delta;
                            this.layers[k][l].dbias += this.layers[k][l].delta;
                            this.layers[i][j].delta = this.layers[k][l].delta * this._sigmoid(this.layers[i][j].sum);
                        }
                    }
                }
            }
        }
    }

    update() {
        let i, j, k;

        for (i = 0; i < this.layers.length; i++) {
            for (j = 0; j < this.layers[i].length; j++) {
                for (k = 0; k < this.layers[i][j].dweights.length; k++) {
                    let pretmp = parseFloat(this.learning_rate * this.layers[i][j].dweights[k]);
                    let tmp = parseFloat(this.layers[i][j].weights[k] - pretmp);
                    this.layers[i][j].weights[k] = tmp;
                }
                this.layers[i][j].bias = (this.layers[i][j].bias - (this.learning_rate * this.layers[i][j].dbias));
            }
        }


    }

    train(times = 1) {
        for (var i = 0; i < times; i++) {
            this.forward();
            this.backward();
            this.update();
            // console.log(this.layers[this.layers.length - 1][0].output);
        }
    }

    output() {
        var output = {}
        output.inputs = this.inputs;
        output.targets = this.targets;
        output.outputs = this.outputs;
        output.error = this.error;

        console.table(output);
    }

    run() {
        this.forward();
        this.output();
    }
}

function iniciar() {


    var net = new MedNet([
        { inputs: [0, 0], outputs: [1] },
        { inputs: [1, 0], outputs: [1] },
        { inputs: [0, 1], outputs: [1] },
        { inputs: [1, 1], outputs: [0] },
    ], [3], 0.15);

    net.train(500);
    net.run();

    //console.log(JSON.stringify(net));

}

iniciar();