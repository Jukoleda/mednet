window.onload = iniciar;

// evals = [{inputs: [0,0], output: 0}]
this.sigmoid = (x) => { return 1 / (1 + Math.exp(-x)) };
this._sigmoid = (x) => { return this.sigmoid(x) * (1 - this.sigmoid(x)) };

class MedNet {
     constructor(structure = [2, 3, 2, 1], learning_rate = 0.15){

          this.layers = new Array();

          this.learning_rate = learning_rate;

          var i,j;

          //inputs
          for(i = 0; i < structure.length; i ++){
               
               this.layers[i] = new Array();

               for(j = 0; j < structure[i]; j++){
                    this.layers[i][j] = {bias: Math.random(), weights: [], otuput:0, sum: 0, dbias: 0, dweights: [], doutput: 0, dsum: 0};
               }
          }
          
     }

     setData(dataset = [{inputs: [0, 0], outputs: [0]}]){
          this.inputs = new Array();
          this.targets = new Array();

          dataset.forEach((item, index) => {
               this.inputs[index] = item.inputs;
               this.targets[index] = item.outputs;
          });
     }

     forward(){
          var i,j,k,context, ik;
          ik = this.inputs;
          context = this;

          this.inputs.forEach((item) => {

               
               for(i = 0; i < context.layers[0].length; i++){
                    context.layers[0][i].sum = context.layers[0][i].bias;
                    context.layers[0][i].sum = context.layers[0][i].sum + (context.layers[0][j].weights[0] * item[i]);
                    context.layers[0][i].output = (1 / (1 + Math.exp( - context.layers[i][j].sum)));
               }

               for(i = 1; i < context.layers.length; i++){
                    for(j = 0; j < context.layers[i].length; j++){
                         context.layers[i][j].sum = context.layers[i][j].bias;
                         for(k = 0; k < context.layers[i-1]; k++){
                              context.layers[i][j].sum = context.layers[i][j].sum + (context.layers[i-1][k].output * context.layers[i][j].weights[k]);
                         }
                         context.layers[i][j].otuput = (1 / (1 + Math.exp( - context.layers[i][j].sum)));
                    }
               }
          });
     }

     backward(){
          var i,j,k,context, ik, lc;
          ik = this.inputs;
          context = this;
          lc = context.layers[context.layers.length - 1];

          this.outputs.forEach((item) => {
               
               for(i = 0; i < context.layers[context.layers.length - 1].length; i++){
                    //dsum es delta
                    //delta =  output - o1 * o1 * (1 - o1);

                    context.layers[context.layers.length - 1][i].dsum = (context.layers[context.layers.length - 1][i].output - item[i]) * context.layers[context.layers.length - 1][i].output * (1 - context.layers[context.layers.length - 1][i].output);
               }

               for(i = context.layers.length - 2; i > 0; i--){
                    for(j = 0; j < context.layers[i].length; j++){
                         for(k = 0; k < context.layers[i][j].dweights.length; k++){

                              context.layers[i+1][j].dweights[k] = context.layers[i+1][j].dsum * context.layers[i][j].output;
                              context.layers[i][j].doutput = context.layers[i+1][j].weights[k] * context.layers[i+1][j].dsum;

                         }
                         context.layers[i][j].dbias = context.layers[i][j].dsum;

                         context.layers[i][j].dsum = context.layers[i][j].doutput;

                        //se multiplica delta * output
                        //dw = output * delta
                        //doutput = peso * delta
                        //dbias  = delta
                    
                        //var h1_delta = o1_delta * (output * (1 - output))

                    }
               }

          });
     }

     update(){
          var i,j,k;

          for(i = 0; i < this.layers.length; i++){
               for(j = 0; j < this.layers[i].length; j++){
                    for(k = 0; k < this.layers[i][j].weights.length; k++){
                         this.layers[i][j].weights[k] = this.layers[i][j].weights[k] - (this.learning_rate * this.layers[i][j].dweights[k]);
                    }
                    this.layers[i][j].bias = this.layers[i][j].bias - (this.learning_rate * this.layers[i][j].dbias);
               }
          }


     }
}
/*

    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {           

        lay[num_layers-1][j].delta = (lay[num_layers-1][j].output - desired_outputs[p][j]) * (lay[num_layers-1][j].output) * (1- lay[num_layers-1][j].output);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {   
            lay[num_layers-2][k].dweights[j] = (lay[num_layers-1][j].delta * lay[num_layers-2][k].output);
            lay[num_layers-2][k].doutput = lay[num_layers-2][k].weights[j] * lay[num_layers-1][j].delta;
        }
            
        lay[num_layers-1][j].dbias = lay[num_layers-1][j].delta;           
    }

    // Hidden Layers
    for(i=num_layers-2;i>0;i--)
    {
        for(j=0;j<num_neurons[i];j++)
        {
             lay[i][j].delta = lay[i][j].doutput;
           
            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i-1][k].dweights[j] = lay[i][j].delta * lay[i-1][k].output;    
                
                if(i>1)
                {
                    lay[i-1][k].doutput = lay[i-1][k].weights[j] * lay[i][j].delta;
                }
            }

            lay[i][j].dbias = lay[i][j].delta;
        }
    }


for (var { input: [i1, i2, i3], output } of data) {

     var h1_inputs =
         pesos.i1_h1 * i1 +

 function iniciar() {

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
     //sum
     var o1_inputs =
         pesos.h1_o1 * h1 +
         pesos.h2_o1 * h2 +
         pesos.h3_o1 * h3 +
         pesos.bias_o1;
         //output
     var o1 = sigmoid(o1_inputs);
     //dsum
     var o1_delta =  output - o1 * sigmoid(x) * (1 - sigmoid(x));


     weight_deltas.h1_o1 += h1 * o1_delta;
     weight_deltas.h2_o1 += h2 * o1_delta;
     weight_deltas.h3_o1 += h3 * o1_delta;
     weight_deltas.bias_o1 += o1_delta;

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

void update_weights(void)
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (alpha * lay[i].neu[j].dw[k]);
            }
            
            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (alpha * lay[i].neu[j].dbias);
        }
    }   
}

void update_weights(void)
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (alpha * lay[i].neu[j].dw[k]);
            }
            
            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (alpha * lay[i].neu[j].dbias);
        }
    }   
}
*/
function iniciar() {
    
    /*var net = new MedNet(2, 1, 1, {input: [1, 1, 1], output: 1});
    net.train(1);
    net.show();*/

    var net = new MedNet();
    
}