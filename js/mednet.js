window.onload = iniciar;


function iniciar() {

    function sigmoid(x) { return 1 / (1 + Math.exp(-x)) } // f(x)
    function _sigmoid(x) { return sigmoid(x) * (1 - sigmoid(x)) } // f'(x)


    var data = [
        { input: [1, 0, 0], output: 1 },
        { input: [0.7, 0, 0], output: 1 }
    ];
    var pesos = {
        i1_h1: Math.random(),
        i2_h1: Math.random(),
        i3_h1: Math.random(),
        bias_h1: Math.random(),

        i1_h2: Math.random(),
        i2_h2: Math.random(),
        i3_h2: Math.random(),
        bias_h2: Math.random(),

        i1_h3: Math.random(),
        i2_h3: Math.random(),
        i3_h3: Math.random(),
        bias_h3: Math.random(),

        h1_o1: Math.random(),
        h2_o1: Math.random(),
        h3_o1: Math.random(),
        bias_o1: Math.random(),

    };



    function nn(i1, i2, i3) {

        var h1_inputs =
            pesos.i1_h1 * i1 +
            pesos.i2_h1 * i2 +
            pesos.i3_h1 * i3 +
            pesos.bias_h1;

        var h1 = sigmoid(h1_inputs);

        var h2_inputs =
            pesos.i1_h2 * i1 +
            pesos.i2_h2 * i2 +
            pesos.i3_h2 * i3 +
            pesos.bias_h2;

        var h2 = sigmoid(h2_inputs);

        var h3_inputs =
            pesos.i1_h3 * i1 +
            pesos.i2_h3 * i2 +
            pesos.i3_h3 * i3 +
            pesos.bias_h3;

        var h3 = sigmoid(h3_inputs);

        var o1_inputs =
            pesos.h1_o1 * h1 +
            pesos.h2_o1 * h2 +
            pesos.h3_o1 * h3 +
            pesos.bias_o1;

        var o1 = sigmoid(o1_inputs);


        return o1;
    }






    var outputResults = () =>
        data.forEach(({ input: [i1, i2, i3], output: y }) =>
            console.log(`[R:${i1}, G:${i2}, B:${i3}] => ${nn(i1, i2, i3)} (expected ${y})`));



    var train = () => {
        const weight_deltas = {
            i1_h1: 0,
            i2_h1: 0,
            i3_h1: 0,
            bias_h1: 0,
            i1_h2: 0,
            i2_h2: 0,
            i3_h2: 0,
            bias_h2: 0,
            i1_h3: 0,
            i2_h3: 0,
            i3_h3: 0,
            bias_h3: 0,
            h1_o1: 0,
            h2_o1: 0,
            h3_o1: 0,
            bias_o1: 0,
        };

        for (var { input: [i1, i2, i3], output } of data) {
            //this part is 100% identic to forward pass function
            var h1_inputs =
            pesos.i1_h1 * i1 +
            pesos.i2_h1 * i2 +
            pesos.i3_h1 * i3 +
            pesos.bias_h1;

        var h1 = sigmoid(h1_inputs);

        var h2_inputs =
            pesos.i1_h2 * i1 +
            pesos.i2_h2 * i2 +
            pesos.i3_h2 * i3 +
            pesos.bias_h2;

        var h2 = sigmoid(h2_inputs);

        var h3_inputs =
            pesos.i1_h3 * i1 +
            pesos.i2_h3 * i2 +
            pesos.i3_h3 * i3 +
            pesos.bias_h3;

        var h3 = sigmoid(h3_inputs);

        var o1_inputs =
            pesos.h1_o1 * h1 +
            pesos.h2_o1 * h2 +
            pesos.h3_o1 * h3 +
            pesos.bias_o1;

        var o1 = sigmoid(o1_inputs);

            //learning starts here:
            // we calculate our delta
            var delta = output - o1;
            //then we calculate our derivative (and throwing away "2 * " as we can multiply it later)
            var o1_delta = delta * _sigmoid(o1_inputs);

            //and for our equatation w1 * h1 + w2 * h2 we're trying to alter weights first

            weight_deltas.h1_o1 += h1 * o1_delta;
            weight_deltas.h2_o1 += h2 * o1_delta;
            weight_deltas.h3_o1 += h3 * o1_delta;
            weight_deltas.bias_o1 += o1_delta;

            //and then we're trying to alter our h1 and h2.
            //but we cannot alter them directly, as they are functions of other weights too
            //so we need to alter their weights by same approach 

            var h1_delta = o1_delta * _sigmoid(h1_inputs);
            var h2_delta = o1_delta * _sigmoid(h2_inputs);
            var h3_delta = o1_delta * _sigmoid(h3_inputs);

            weight_deltas.i1_h1 += i1 * h1_delta;
            weight_deltas.i2_h1 += i2 * h1_delta;
            weight_deltas.i3_h1 += i3 * h1_delta;
            weight_deltas.bias_h1 += h1_delta;

            weight_deltas.i1_h2 += i1 * h2_delta;
            weight_deltas.i2_h2 += i2 * h2_delta;
            weight_deltas.i3_h2 += i3 * h2_delta;
            weight_deltas.bias_h2 += h2_delta;

            weight_deltas.i1_h3 += i1 * h3_delta;
            weight_deltas.i2_h3 += i2 * h3_delta;
            weight_deltas.i3_h3 += i3 * h3_delta;
            weight_deltas.bias_h3 += h3_delta;
        }

        return weight_deltas;
    }

    var applyTrainUpdate = (weight_deltas = train()) =>
        Object.keys(pesos).forEach(key =>
            pesos[key] += weight_deltas[key]);

    for (var i = 0; i < 1000; i++) {
        applyTrainUpdate();
    }
    console.log(pesos);
    outputResults();


}