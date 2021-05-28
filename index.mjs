// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import '@tensorflow/tfjs-node';

import * as canvas1 from 'canvas';
let canvas = canvas1['default']
import * as faceapi from 'face-api.js';

const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })
await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models')
await faceapi.nets.faceLandmark68Net.loadFromDisk('./models')
await faceapi.nets.faceExpressionNet.loadFromDisk('./models')
await faceapi.nets.faceRecognitionNet.loadFromDisk('./models')

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      resolve(img)
    }
    img.onerror = err => { reject(err) }
    img.src = src
  })
}

async function image2descriptor(data) {
  let result = await faceapi
    .detectSingleFace(await loadImage(data))
    .withFaceLandmarks()
    .withFaceDescriptor()
  return result.descriptor;
}


var service = protoLoader.loadSync(
  "face.proto",
  {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
  })['face.FaceMatch'];
async function detect(call, callback) {
  try {
    let points =  Array.from(await image2descriptor(call.request.data))
    console.log(points)
    callback(null, { points });
  } catch (error) {
    callback(error, null);
  }
}

async function matchDescriptors(call, callback) {
  try {
    let { a, b } = call.request;
    let faceMatcher = new faceapi.FaceMatcher([a])
    let { distance } = faceMatcher.findBestMatch(b)
    callback(null, { distance });
  } catch (error) {
    callback(error, null);
  }
}
async function matchDescriptorToImage(call, callback) {
  try {
    console.log(call.request)
    let { image, descriptor } = call.request;
    let descriptor2 = await image2descriptor(image.data);
    let faceMatcher = new faceapi.FaceMatcher([new Float32Array(descriptor.points)])
    let { distance } = faceMatcher.findBestMatch(descriptor2)
    callback(null, { distance });
  } catch (error) {
    callback(error, null);
  }
}
var server = new grpc.Server();
server.addService(service, {
  detect: detect,
  matchDescriptorToImage: matchDescriptorToImage,
  matchDescriptors: matchDescriptors
});
server.bindAsync('0.0.0.0:50051', grpc.ServerCredentials.createInsecure(), () => {
  server.start();
});
