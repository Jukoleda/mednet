window.onload = iniciar;

// evals = [{inputs: [0,0], output: 0}]

class MedNet {
     constructor(structure = [2, 3, 2, 1]){

          this.layers = new Array();

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
          lc = this.layers[this.layers.length - 1];

          this.outputs.forEach((item) => {
               
               for(i = 0; i < lc.length; i++){
                    lc[i].dsum = (lc[i].output - item[i]) * lc[i].output * (1 - lc[i].output);
               }

               for(i = context.layers.length - 2; i > 0; i--){
                    for(j = 0; j < context.layers[i].length; j++){
                        // context.layers[i][j].doutput = context.layers[i][j].doutput
                    }
               }

          });
     }
}



function iniciar() {
    
    /*var net = new MedNet(2, 1, 1, {input: [1, 1, 1], output: 1});
    net.train(1);
    net.show();*/

    var net = new MedNet();
    
}