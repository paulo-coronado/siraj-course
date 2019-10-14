const MODEL_URL = '../static/model_web/'
const LABELS_URL = MODEL_URL + 'labels.json'
const MODEL_JSON = MODEL_URL + 'model.json'

// SECTION Global variables initializer
// Bounding box features relevant to fall detection
var lf = 0 // Laid frames (width > height)

var frames = 1
var fps = 6

var isPast = true
var isIgnored = false
var waitTimeOut = false

var pastW = [] // Width values BEFORE a possible fall
var pastH = [] // Height values BEFORE a possible fall
var futW = [] // Width values AFTER a possible fall
var futH = [] // Height values AFTER a possible fall

var lifeW = [] // pastW + futW
var lifeH = [] // pastH + futH

var P = [] // Array to calculate the max and min proportion (H / W)

var deltaW = [] // Width delta (n + 1 box - n)
var deltaH = [] // Height delta (n + 1 box - n)
var deltaP = [] // Proportion delta (n + 1 box - n)

var maxW = 0 // Max box width in a fall 
var maxH = 0 // Max box height in a fall 
var maxP = 0 // Max box proportion in a fall 

var minW = 0 // Min box width in a fall 
var minH = 0 // Min box height in a fall 
var minP = 0 // Min box proportion in a fall 

var avgW = 0 // Avg box width in a fall
var avgH = 0 // Avg box height in a fall
var avgP = 0 // Avg box proportion in a fall

var maxDeltaW = 0 // Max width delta (n + 1 box - n)
var maxDeltaH = 0 // Max height delta (n + 1 box - n)
var maxDeltaP = 0 // Max proportion delta (n + 1 box - n)

$('document').ready(() => {
  var videoRef = $('#video')
  var canvasRef = $('#canvas')

  $("#loading").delay(4000).hide(0)
  $("#video-player").delay(4000).fadeIn()
  builder(videoRef, canvasRef)

})

const TFWrapper = model => {
  const calculateMaxScores = (scores, numBoxes, numClasses) => {
    const maxes = []
    const classes = []
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE
      let index = -1
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j]
          index = j
        }
      }
      maxes[i] = max
      classes[i] = index
    }
    return [maxes, classes]
  }

  const buildDetectedObjects = (
    width,
    height,
    boxes,
    scores,
    indexes,
    classes
  ) => {
    const count = indexes.length
    const objects = []
    for (let i = 0; i < count; i++) {
      const bbox = []
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j]
      }
      const minY = bbox[0] * height
      const minX = bbox[1] * width
      const maxY = bbox[2] * height
      const maxX = bbox[3] * width
      bbox[0] = minX
      bbox[1] = minY
      bbox[2] = maxX - minX
      bbox[3] = maxY - minY
      objects.push({
        bbox: bbox,
        class: classes[indexes[i]],
        score: scores[indexes[i]]
      })
    }
    return objects
  }

  const detect = input => {
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(input)
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0)
    })

    const height = batched.shape[1]
    const width = batched.shape[2]

    return model.executeAsync(batched).then(result => {
      const scores = result[0].dataSync()
      const boxes = result[1].dataSync()

      // clean the webgl tensors
      batched.dispose()
      tf.dispose(result)

      const [maxScores, classes] = calculateMaxScores(
        scores,
        result[0].shape[1],
        result[0].shape[2]
      )

      const prevBackend = tf.getBackend()
      // run post process in cpu
      tf.setBackend('cpu')
      const indexTensor = tf.tidy(() => {
        const boxes2 = tf.tensor2d(boxes, [
          result[1].shape[1],
          result[1].shape[3]
        ])
        return tf.image.nonMaxSuppression(
          boxes2,
          maxScores,
          20, // maxNumBoxes
          0.5, // iou_threshold
          0.5 // score_threshold
        )
      })
      const indexes = indexTensor.dataSync()
      indexTensor.dispose()
      // restore previous backend
      tf.setBackend(prevBackend)

      return buildDetectedObjects(
        width,
        height,
        boxes,
        maxScores,
        indexes,
        classes
      )
    })
  }
  return {
    detect: detect
  }
}

const builder = (videoRef, canvasRef) => {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const webCamPromise = navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: 'user'
        }
      })
      .then(stream => {
        window.stream = stream
        videoRef.get(0).srcObject = stream
        return new Promise((resolve, _) => {
          videoRef.get(0).onloadedmetadata = () => {
            resolve()
          }
        })
      })

    const modelPromise = tf.loadGraphModel(MODEL_JSON)
    const labelsPromise = fetch(LABELS_URL).then(data => data.json())
    Promise.all([modelPromise, labelsPromise, webCamPromise])
      .then(values => {
        const [model, labels] = values
        detectFrame(videoRef.get(0), canvasRef.get(0), model, labels)
      })
      .catch(error => {
        console.error(error)
      })
  }

}

const detectFrame = (video, canvas, model, labels) => {
  TFWrapper(model)
    .detect(video)
    .then(predictions => {
      renderPredictions(canvas, predictions, labels)
      requestAnimationFrame(() => {
        detectFrame(video, canvas, model, labels)
      })
    })
}

// NOTE This code runs for every frame
const renderPredictions = (canvas, predictions, labels) => {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  frames += 1

  setTimeout(function () {
    frames -= 1
    fps = frames / 5
  }, 5000)

  // Font options.
  const font = '16px sans-serif'
  ctx.font = font
  ctx.textBaseline = 'top'

  isIgnored = false

  // Clean variables if there is no person OR there are more than one
  if (predictions.length !== 1) {
    isIgnored = true
  }

  // Only runs if an object was detected (forEach)
  predictions.forEach(prediction => {
    const x = prediction.bbox[0]
    const y = prediction.bbox[1]
    const width = prediction.bbox[2]
    const height = prediction.bbox[3]
    const label = labels[parseInt(prediction.class)]

    // Change boxing color 
    if (label === 'danger') {
      isPast = false

      // Draw the bounding box
      ctx.strokeStyle = '#FF0000'

      // Draw the label background
      ctx.fillStyle = '#FF0000'
    } else {
      // Draw the bounding box
      ctx.strokeStyle = '#00FFFF'

      // Draw the label background
      ctx.fillStyle = '#00FFFF'
    }

    if (isPast) {
      if (pastW.length <= (Math.round(fps) * 3)) {
        if (!isIgnored) {
          pastW.push(width.toFixed(4))
          pastH.push(height.toFixed(4))
        } else {
          pastW.push(-1)
          pastH.push(-1)
        }
      } else {
        for (let i = 0; i < (pastW.length - (Math.round(fps)) * 3); i++) {
          pastW.shift()
          pastH.shift()
        }
        if (!isIgnored) {
          pastW.shift()
          pastH.shift()
          pastW.push(width.toFixed(4))
          pastH.push(height.toFixed(4))
        } else {
          pastW.shift()
          pastH.shift()
          pastW.push(-1)
          pastH.push(-1)
        }
      }
    } else {
      if (!isIgnored) {
        futW.push(width.toFixed(4))
        futH.push(height.toFixed(4))
      } else {
        futW.push(-1)
        futH.push(-1)
      }
      // console.log(futW)
      if (!waitTimeOut) {
        setTimeout(function () {

          lifeW = pastW.concat(futW)
          lifeH = pastH.concat(futH)
          // console.log(lifeW)
          // console.log(lifeH)

          // Check if there is more than one detection
          if (!(lifeW.includes(-1) || lifeH.includes(-1))) {
            // SECTION Generate ML features
            // Max Width (x);               Min Width (x);             Avg Width (x);
            // Max Height (x);              Min Height (x);            Avg Height (x); 
            // Max Proportion (x);           Min Proportion (x);         Avg Proportion (x);
            // Max Delta Height (x);  Min Delta Width (x); Max Delta Proportion (x); Laid Frames (x).

            // Laid frames
            lifeW.map(function (item, index) {
              if (item - lifeH[index] > 0) {
                lf++
              }
            })

            // Proportion (H / W)
            P = lifeH.map(function (item, index) {
              return (item / lifeW[index]).toFixed(4)
            })

            // Avg values
            for (var i = 0; i < lifeW.length; i++) {
              avgW += Number(lifeW[i])
              avgH += Number(lifeH[i])
              avgP += Number(P[i])

              if (i > 0) {
                // Delta definition
                deltaW.push((Math.abs(Number(lifeW[i]) - Number(lifeW[i - 1]))).toFixed(4))
                deltaH.push((Math.abs(Number(lifeH[i]) - Number(lifeH[i - 1]))).toFixed(4))
                deltaP.push((Math.abs(Number(P[i]) - Number(P[i - 1]))).toFixed(4))
              }
            }

            // Max, min and avg values
            maxW = Math.max(...lifeW)
            minW = Math.min(...lifeW)
            maxH = Math.max(...lifeH)
            minH = Math.min(...lifeH)
            maxP = Math.max(...P)
            minP = Math.min(...P)
            maxDeltaW = Math.max(...deltaW)
            maxDeltaH = Math.max(...deltaH)
            maxDeltaP = Math.max(...deltaP)
            avgH = Number((avgH / lifeH.length).toFixed(4))
            avgW = Number((avgW / lifeW.length).toFixed(4))
            avgP = Number((avgP / P.length).toFixed(4))

            // NOTE At this moment, there is a full life cycle (past and future)
            // It is time to ML inference
            console.log('############### FEATURES OVERVIEW ###############')
            console.log('MaxW: ', maxW)
            console.log('MinW: ', minW)
            console.log('AvgW: ', avgW)
            console.log('MaxH: ', maxH)
            console.log('MinH: ', minH)
            console.log('AvgH: ', avgH)
            console.log('MaxP: ', maxP)
            console.log('MinP: ', minP)
            console.log('AvgP: ', avgP)
            console.log('MaxDeltaW: ', maxDeltaW)
            console.log('MaxDeltaH: ', maxDeltaH)
            console.log('MaxDeltaP: ', maxDeltaP)
            console.log('Lf: ', lf)
            console.log('Features sent to ML model...')
          }

          // Next detection reset
          pastW = []
          pastH = []
          futW = []
          futH = []
          lifeW = []
          lifeH = []
          avgW = 0
          avgH = 0
          avgP = 0
          deltaW = []
          deltaH = []
          deltaP = []
          P = []
          lf = 0

          isPast = true
          waitTimeOut = false
        }, 3000)
        waitTimeOut = true
      }
    }

    ctx.lineWidth = 4
    ctx.strokeRect(x, y, width, height)

    const textWidth = ctx.measureText(label).width
    const textHeight = parseInt(font, 10) // base 10
    ctx.fillRect(x, y, textWidth + 4, textHeight + 4)
  })

  // NOTE Style for each detection
  predictions.forEach(prediction => {
    const x = prediction.bbox[0]
    const y = prediction.bbox[1]
    const label = labels[parseInt(prediction.class)]
    // Draw the text last to ensure it's on top
    ctx.fillStyle = '#000000'
    ctx.fillText(label, x, y)
  })
}