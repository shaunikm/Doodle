import { useRef, useState, useEffect } from 'react';
import Canvas, { Controls } from './Canvas';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as sketch from '@magenta/sketch';

type Tool = 'draw' | 'erase';

function App() {
  const [currentTool, setCurrentTool] = useState<Tool>('draw');
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const loadSketchRNN = async () => {
      try {
        const model = new sketch.SketchRNN('cat');
        await model.initialize();
        console.log('SketchRNN model loaded');
      } catch (error) {
        console.error('Error loading SketchRNN model:', error);
      }
    };

    loadSketchRNN();
  }, []);

  useEffect(() => {
    const setBackend = async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TensorFlow.js backend set to WebGL');
      } catch (error) {
        console.error('Error setting TensorFlow.js backend:', error);
      }
    };

    setBackend();
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    setPrediction(null);
    setImageDataUrl(null);
  };

  const classifyCanvas = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      // Load the COCO-SSD model
      const model = await cocoSsd.load();

      // Get the image data from the canvas
      const imageData = canvas.toDataURL('image/png');
      setImageDataUrl(imageData);

      const img = new Image();
      img.src = imageData;

      await new Promise((resolve) => {
        img.onload = resolve;
      });

      console.log('Image loaded for classification:', img);

      // Use the model to detect objects in the image
      const predictions = await model.detect(img);
      console.log('Predictions:', predictions);

      if (predictions.length > 0) {
        setPrediction(predictions[0].class);
      } else {
        setPrediction('No prediction');
      }
    } catch (error) {
      console.error('Error using COCO-SSD model:', error);
    }
  };

  return (
    <div className="h-screen overflow-hidden flex flex-col">
      <Canvas
        ref={canvasRef}
        currentTool={currentTool}
        isDrawing={isDrawing}
        setIsDrawing={setIsDrawing}
      />
      <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center border-b border-[#2a2a2a] bg-[#1a1a1a]">
        <h1 className="text-xl font-medium text-[#efefef]">
          Doodle Placeholder Text
        </h1>
        <div className="flex items-center gap-4">
          <Controls
            currentTool={currentTool}
            setCurrentTool={setCurrentTool}
            onClear={clearCanvas}
          />
          <button onClick={classifyCanvas} className="p-2.5 rounded bg-[#42b883] text-white">
            Classify
          </button>
        </div>
      </div>
      {imageDataUrl && (
        <div className="absolute bottom-16 left-0 right-0 p-4 bg-[#1a1a1a] text-[#efefef] text-center">
          <img src={imageDataUrl} alt="Captured canvas" className="max-h-32 mx-auto" />
        </div>
      )}
      {prediction && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-[#1a1a1a] text-[#efefef] text-center">
          Prediction: {prediction}
        </div>
      )}
    </div>
  );
}

export default App;
