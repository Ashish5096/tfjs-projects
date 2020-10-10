const canvas =  document.getElementById("pose-canvas");
const ctx = canvas.getContext("2d");
const video = document.getElementById("pose-video");

const config ={
      video:{ width: 640, height: 480, fps: 50}
    };

function drawPoint(x, y, radius, color)
{
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawLine(x1,y1,x2,y2,color)
{
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.stroke();
}

function draw(part,radius,color)
{
  let k;
    for(k=0; k<part.length-1; k++)
    {
        var[x1,y1] = part[k];
        var[x2,y2] = part[k+1];

        drawPoint(x1,y1, radius,color);
        drawLine(x1,y1,x2,y2,color); 
    }

    [x1,y1] = part[k];
    drawPoint(x1,y1, radius, color);
}


async function estimateHands(model)
{
    ctx.clearRect(0, 0, config.video.width, config.video.height);
    const predictions = await model.estimateHands(video);
    
    if (predictions.length > 0) 
    {
      
      for(let i=0; i<predictions.length;i++)
      {
          const thumb_finger = predictions[i].annotations['thumb'];
          const index_finger = predictions[i].annotations['indexFinger'];
          const middle_finger = predictions[i].annotations['middleFinger'];
          const ring_finger = predictions[i].annotations['ringFinger'];
          const pinky_finger = predictions[i].annotations['pinky'];
          const palm = predictions[i].annotations['palmBase'];

          draw(thumb_finger,3,'red');
          draw(index_finger,3,'red');
          draw(middle_finger,3,'red');
          draw(ring_finger,3,'red');
          draw(pinky_finger,3,'red');

          let[x1,y1] = palm[0];
          drawPoint(x1,y1, 3,'red');

          let[x2,y2] = thumb_finger[0];
          drawLine(x1,y1,x2,y2,'red');

          [x2,y2] = index_finger[0];
          drawLine(x1,y1,x2,y2,'red');

          [x2,y2] = middle_finger[0];
          drawLine(x1,y1,x2,y2,'red');

          [x2,y2] = ring_finger[0];
          drawLine(x1,y1,x2,y2,'red');

          [x2,y2] = pinky_finger[0];
          drawLine(x1,y1,x2,y2,'red');

        } 
    }
    setTimeout(function(){
      estimateHands(model);
    }, 1000 / config.video.fps)
}

async function main()
{
    //await tf.setBackend('wasm');
    const model = await handpose.load();
    estimateHands(model);
    console.log("Starting predictions")  
}

async function init_camera()
{
    const constraints ={
      audio: false,
      video:{
      width: config.video.width,
      height: config.video.height,
      frameRate: { max: config.video.fps }
      }
    };

    video.width = config.video.width;
    video.height= config.video.height;
    canvas.width = config.video.width;
    canvas.height = config.video.height;
   
    console.log("Canvas initialized");

    navigator.mediaDevices.getUserMedia(constraints).then(stream => {
       video.srcObject = stream;
        main();
    });
}



document.addEventListener('DOMContentLoaded',function(){
  init_camera();
});