window.onload = iniciar;



class MedNode {
     constructor(bias = Math.random()){
         this.bias = bias;
         this.output = 0;
         this.delta = 0;
         this.default = {
              input: 0,
              weight: Math.random(),
              output: 0,
              delta: 0
         };
         this.inputs = Array();
         this.weights = Array();
     }
     
     sigmoid(x){
          return 1 / (1 + Math.exp(-x));
     }

     _sigmoid(x) { 
          return this.sigmoid(x) * (1 - this.sigmoid(x));
     }



     addInput(input, id) {
          this.inputs.push(input);
          this.weights.push(this.default.weight);
     }
     addInput(input) {
          this.inputs.push(input);
          this.weights.push(this.default.weight);
     }

     activate() {

          this.output = 0;

          for(var i = 0; i < this.inputs.length; i++)
               this.output += this.inputs[i] * this.weights[i];
          
          this.output += this.bias; 

          this.output = this.sigmoid(this.output);
     }

     _activate(target) {

          var activation = 0;
          var delta = 0;
          
          for(var i = 0; i < this.inputs.length; i++)
               activation += this.inputs[i] * this.weights[i];
          activation += this.bias; 

          delta = target - this.sigmoid(activation);
          delta = delta * this._sigmoid(delta);

          for(var i = 0; i < this.inputs.length; i++)
               this.weights[i] += this.inputs[i] * delta;
          
          this.bias += delta;
          this.delta = delta;   
     }

     debug() {
          var percep = {};
          percep.inputs = this.inputs;
          percep.weights = this.weights;
          percep.bias = this.bias;
          console.table(percep);
     }

 }
 

class MedNet {
     constructor(hidden, deep, output, dataset = {input: [], output: 0}){
          this.inputs = Array();
          this.hidden = Array();
          this.outputs = Array();
          this.dataset = dataset;

          for(var i = 0; i < this.dataset.input.length; i ++){
               this.inputs.push(new MedNode());
               this.inputs[i].addInput(this.dataset.input[i]);
          }
          
          for(var i = 0; i < deep; i++){
               var tmp = Array();
               for(var j = 0; j < hidden; j++){
                    tmp.push(new MedNode());
               }
               this.hidden.push(tmp);
          }
          for(var i = 0; i < output; i ++){
               this.outputs.push(new MedNode());
          }

     }

     forward() {
          this.inputs.forEach((item) => item.activate());
          this.hidden[0].forEach((hidden) => 
               this.inputs.forEach((input) => hidden.addInput(input.output)));
          for(var i = 1; i < this.hidden.length; i++){
               for(var j = 0; j < this.hidden[i].length; j++){
                    this.hidden[i-1][j].activate();
                    for(var k = 0; k < this.hidden[i].length; k++)
                         this.hidden[i][k].addInput(this.hidden[i-1][j].output);
               }
          }
          this.hidden[this.hidden.length -1].forEach((item) => this.outputs.forEach((out) =>{
               item.activate();
               out.addInput(item.output);
          }));
          
          this.outputs.forEach((item) => item.activate());
     }
     
     backward(target) {
          this.outputs.forEach((item) => item.activate(target));
          
          if(this.hidden.length > 1){
               this.hidden[this.hidden.length -1].forEach(
                    (item) => this.outputs.forEach(
                         (out) => item._activate(out.delta)
                         ));
               
               for(var i = this.hidden.length - 2; i > 0; i--){
                    for(var j = this.hidden[i].length; j > 0; j--){
                         for(var k = this.hidden[i].length; k > 0; k--){
                              this.hidden[i][j]._activate(this.hidden[i+1][k].delta);
                         }
                    }
               }
          }else{
               this.hidden[0].forEach((hidden) => {
                    this.outputs.forEach((output) => {
                         hidden._activate(output.delta);
                    })
               });
          }
          this.hidden[0].forEach((hidden) => this.inputs.forEach((input) => input._activate(hidden.delta)));
          
     }

     train(times = 1){
          for(var i = 0; i < times; i++){
               this.forward();
          
               this.backward(this.dataset.output);
          }
     }

     setConfig(dataset = {weights: {inputs: [], hidden:[[]], outputs: []}, bias: {inputs: [], hidden:[[]], outputs: []}}) {

        for(var i = 0; i < this.inputs.length; i++){
            this.inputs[i].weight = weights.inputs[i];
            this.inputs[i].bias = bias.inputs[i];
        }
        
        for(var i = 0; i < this.hidden.length; i++){
            for(var j = 0; j < this.hidden[i].length; j++){
                this.hidden[i][j].weight = weights.hidden[i][j];
                this.hidden[i][j].bias = bias.hidden[i][j];
            }
        }
        
        for(var i = 0; i < this.outputs.length; i++){
            this.outputs[i].weight = weights.outputs[i];
            this.outputs[i].bias = bias.outputs[i];
        }

     }

     show(){
          console.table(this.dataset);
          console.log("inputs");
          console.table(this.inputs);
          console.log("hidden");
          console.table(this.hidden[0]);
          console.log("outputs");
          console.table(this.outputs);
     }
    
    getData(){
        data = {};
        data.weights = {};
        data.bias = {};
        data.weights.inputs = Array();
        data.weights.hidden = Array();
        data.weights.outputs = Array();
        data.bias.outputs = Array();
        data.bias.outputs = Array();
        data.bias.outputs = Array();

        this.inputs.forEach((item) => {
            data.weights.inputs.push(item.weight);
            data.bias.inputs.push(item.bias);
        });

        this.hidden.forEach((l) => {
            var temp = Array();
            var temp2 = Array();
            l.forEach((p) => {
                temp.push(p.weight);
                temp2.push(p.bias);
            });
            data.weights.hidden.push(temp);
            data.bias.hidden.push(temp);
        });

        this.outputs.forEach((item) => {
            data.weights.outputs.push(item.weight);
            data.bias.outputs.push(item.bias);
        });
    
        return data;
        
    }
}




function iniciar() {
    
    var net = new MedNet(2, 1, 1, {input: [1, 1, 1], output: 1});
    net.train(1000);
    net.show();

    
}