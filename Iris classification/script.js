
const IRIS_CLASSES =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
const IRIS_NUM_CLASSES = IRIS_CLASSES.length;
const url = 'http://127.0.0.1:8080/';
let model=null;



async function getData()
{
	const DataReq = await fetch('iris.json');
    const Data = await DataReq.json();

    const values = Data.map(item => ({ 
        a: item.sepalLength, b: item.sepalWidth,
        c: item.petalLength, d: item.petalWidth,
        label: item.species
    }));
    
    const FilteredValues = values.filter(item => (

        item.a != null && item.b != null && item.c != null && item.d != null && item.label != null 
    ));
    
    tf.dispose([values, Data, DataReq]);

	return FilteredValues;
}


function convertToTensor(dataset)
{ 

  return tf.tidy(() => {

    tf.util.shuffle(dataset);
    tf.util.shuffle(dataset);

    const inputs = dataset.map(item => [item.a, item.b, item.c, item.d])
    
    const labels=[];
    for(i=0;i<dataset.length;i++)
    {
      if(dataset[i].label == 'setosa')
        labels.push(0)
      else if(dataset[i].label == 'versicolor')
        labels.push(1)
      else if(dataset[i].label == 'virginica')
        labels.push(2)
    }
  
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 4]);
    const labelTensor = tf.oneHot(tf.tensor1d(labels).toInt(), IRIS_NUM_CLASSES);

    return {
      inputs: inputTensor,
      labels: labelTensor
   }
 });

}


async function run() 
{
    let value=0;
    const data = await getData();
    const tensorData =  convertToTensor(data);
    const {inputs, labels} = tensorData;

    tf.dispose(data);
      
    document.getElementById("btn1").onclick = function() { startTrain(inputs,labels)};

    document.getElementById("pl1").onclick = function() { 
        value = parseFloat(document.getElementById('plength').value) ;
        value = (value + 0.1).toFixed(1);
        document.getElementById('plength').value = value;
        testSample()
    };
    document.getElementById("pl2").onclick = function() { 
        value = parseFloat(document.getElementById('plength').value) ;
        value = (value -0.1).toFixed(1);
        document.getElementById('plength').value = value;
        testSample() 
    };


    document.getElementById("pw1").onclick = function() {
        value = parseFloat(document.getElementById('pwidth').value) ;
        value = (value + 0.1).toFixed(1);
        document.getElementById('pwidth').value = value;
        testSample() 
    };
    document.getElementById("pw2").onclick = function() { 
        value = parseFloat(document.getElementById('pwidth').value);
        value = (value - 0.1).toFixed(1);
        document.getElementById('pwidth').value = value;
        testSample() 
    };

    document.getElementById("sl1").onclick = function() { 
        value = parseFloat(document.getElementById('slength').value);
        value = (value + 0.1).toFixed(1);
        document.getElementById('slength').value = value;
        testSample() 
    };
    document.getElementById("sl2").onclick = function() { 
        value = parseFloat(document.getElementById('slength').value);
        value = (value - 0.1).toFixed(1);
        document.getElementById('slength').value = value;
        testSample() 
    };

    document.getElementById("sw1").onclick = function() {
        value = parseFloat(document.getElementById('swidth').value);
        value = (value + 0.1).toFixed(1);
        document.getElementById('swidth').value = value; 
        testSample() 
    };
    document.getElementById("sw2").onclick = function() { 
        value = parseFloat(document.getElementById('swidth').value);
        value = (value - 0.1).toFixed(1);
        document.getElementById('swidth').value = value; 
        testSample() 
    };

    document.getElementById("btn2").onclick = function() { loadHostedModel()};
    document.getElementById("btn3").onclick = function() { saveModel()};
    document.getElementById("btn4").onclick = function() { loadLocalModel()};
}


async function startTrain(data_x,data_y)
{
    const summaryContainer = document.getElementById('summ-block');
    const epoch =  document.getElementById("epoch").value;
    const learning_rate = document.getElementById("lrate").value;
    const validationSplit =0.3;

    const val_len = validationSplit * data_x.shape[0]
    const train_len = data_x.shape[0] -val_len

    const [train_x, val_x] = tf.split(data_x, [train_len, val_len], 0);
    const [train_y, val_y] = tf.split(data_y, [train_len, val_len], 0);

    if(epoch >0 && learning_rate >0)
    {
        if(model == null)
        {
          model = createModel()
        }
        tfvis.show.modelSummary(summaryContainer, model);
        await trainModel(epoch, learning_rate, train_x, train_y, val_x, val_y); 
        testModel(val_x, val_y)
        document.getElementById("btn3").disabled = false;       
    }
}

async function trainModel(epoch_val,learning_rate, train_x, train_y, val_x, val_y)
{
    model.compile({
      optimizer: tf.train.adam(learningRate = learning_rate),
      loss: tf.losses.softmaxCrossEntropy,
      metrics: ['accuracy'],
    });

    const lossContainer = document.getElementById('loss-block');
    const accContainer = document.getElementById('acc-block');
    const batchSize = 32;
    const epochs = epoch_val;
    const history = [];
    const beginMs = performance.now();
    
    await model.fit(train_x, train_y, {
      batchSize,
      epochs,
      validationData: [val_x,val_y],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) => {

          const secPerEpoch =(performance.now() - beginMs) / (1000 * (epoch + 1));
          document.getElementById("status").innerHTML = "Training model approx "+secPerEpoch.toFixed(4)+" seconds per epoch";
          
          history.push(log);
          tfvis.show.history(accContainer, history, ['acc', 'val_acc'], { height:200 });
          tfvis.show.history(lossContainer, history, ['loss', 'val_loss'],{ height:200 });
          createConfusionMatrix(model,val_x,val_y);
        }
      }
    });
}


function createModel()
{
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [4], units: 50, useBias: true, activation:'relu'}));
  model.add(tf.layers.dense({units: 20, activation: 'relu'}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  return model;
}

function maxIndex(arr)
{
  let max=0, index=0;

  for(k=0;k<arr.length;k++)
  {
    if(arr[k] > max)
    {
      max = arr[k];
      index = k;
    }   
  }

  return index;
}
function testModel(data_x, data_y)
{
    const predictions = model.predict(data_x).arraySync()
    const true_labels  = data_y.argMax(axis=1).arraySync();
    const features = data_x.arraySync();
    let table = document.getElementById("myTable");
    let prob,row,cell, predicted_label;
    
    for(i=0;i<predictions.length;i++)
    {
      prob = predictions[i].map(item => Number(item.toFixed(2)) )
      predicted_label = maxIndex(prob);

      row = table.insertRow(2);
      for(j=0;j< 4;j++)
      {
          cell = row.insertCell(j);
          cell.innerHTML = features[i][j].toFixed(1);
      }
    
      cell = row.insertCell(4);
      cell.innerHTML = IRIS_CLASSES[true_labels[i]];

      cell = row.insertCell(5);
      cell.innerHTML = IRIS_CLASSES[predicted_label];

      if(true_labels[i] == predicted_label)
      {
        cell.style.backgroundColor = "rgb(154,205,50)";
      }
      else
      {
        cell.style.backgroundColor = "rgb(255,102,102)";
      }
        
      cell = row.insertCell(6);
      cell.innerHTML = prob;
    }
}

function testSample()
{
    let pl = parseFloat(document.getElementById('plength').value);
    let pw = parseFloat(document.getElementById('pwidth').value);
    let sl = parseFloat(document.getElementById('slength').value);
    let sw = parseFloat(document.getElementById('swidth').value);
    
    const x = tf.tensor2d([pl, pw, sl, sw], [1, 4]);

    const ys = model.predict(x);

    let y = tf.squeeze(ys);
    y = y.arraySync();
    y = y.map(item => Number(item.toFixed(2)) );

    let i = maxIndex(y)

    document.getElementById('pclass').innerHTML = IRIS_CLASSES[i];
    document.getElementById('pprob').innerHTML = y;   
}

async function saveModel()
{
  await model.save('downloads://my-model');

  document.getElementById("status").innerHTML = "Model Downloaded Succesfully!!!"; 
}

async function loadLocalModel()
{
   model = await tf.loadLayersModel(url+'Iris_classification/my-model.json');

   //model = await tf.loadGraphModel(url+'Iris_classification/my-model.weights.bin');
   document.getElementById("status").innerHTML = "Model Loaded Succesfully !!!"; 
}

async function loadHostedModel()
{
  const HOSTED_MODEL_JSON_URL ='https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';
  model = await tf.loadLayersModel(HOSTED_MODEL_JSON_URL);

  document.getElementById("btn3").disabled = false;
  document.getElementById("status").innerHTML = "Model Loaded Succesfully !!!";   
}

async function createConfusionMatrix(Modeldata,val_x,val_y)
{
    const matrxiContainer = document.getElementById('matrix-block');
    const predictions = Modeldata.predict(val_x).argMax(axis=1);
    const labels = val_y.argMax(axis=1)

    const matrixData = await tfvis.metrics.confusionMatrix(labels, predictions);

    tfvis.render.confusionMatrix(
        matrxiContainer,
        {values: matrixData, tickLabels:IRIS_CLASSES},
        {shadeDiagonal: true, height: 200 }
    );

    tf.dispose([predictions, labels]);
}

document.addEventListener('DOMContentLoaded', run);