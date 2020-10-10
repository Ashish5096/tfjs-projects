let classifier;
let video;
let label= '';
let prob = 0;

function modelReady()
{
	console.log("Model Ready !!!");
	classifier.predict(gotResults);
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
        label = results[0].label;
        prob = results[0].confidence;
        classifier.predict(gotResults);
	}		
}

function draw()
{
    background(0)
    image(video,0,0);
    fill(255)                       // font color of text
	textSize(24)                    // font size of text
    text(label+"   "+prob ,10,height-20)
}

function setup()
{
	createCanvas(640,520);
    video = createCapture(VIDEO)
    video.hide()
	classifier = ml5.imageClassifier('MobileNet',video,modelReady)
}
