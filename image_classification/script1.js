let classifier;
let img;

function modelReady()
{
	console.log("Model Ready !!!");
	classifier.predict(img,gotResults);
}

function gotResults(error, results)
{
	if(error)
	{
		console.error(error)
	}
	else
	{
		console.log(results)
	}

	let name = results[0].label;
	let prob = results[0].confidence;

	fill(0)
	textSize(64)
	text(name ,10,height-100)	
	createP(name)
	createP(prob)
}

function imageReady()
{
	image(img,0,0,width,height);
}

function setup()
{
	createCanvas(640,480);
	img = createImg("images/puffin.jpeg","puffin.jpeg","",callbacks=imageReady)
	img.hide()
	background(0);
	classifier = ml5.imageClassifier('MobileNet',modelReady)
}
