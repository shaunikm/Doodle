import { useRef, useState, useEffect } from 'react';
import Canvas, { Controls } from './Canvas';
import * as tf from '@tensorflow/tfjs';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';

type Tool = 'draw' | 'erase';

class L2Regularizer extends Regularizer {
  static className = 'L2Regularizer';
  private l2: number;

  constructor(config: { l2: number }) {
    super();
    this.l2 = config.l2;
  }

  apply(x: tf.Tensor): tf.Scalar {
    return tf.tidy(() => {
      const regularization = tf.mul(this.l2, tf.sum(tf.square(x)));
      return tf.scalar(regularization.arraySync() as number);
    });
  }

  getConfig() {
    return { l2: this.l2 };
  }
}

tf.serialization.registerClass(L2Regularizer);

function App() {
  const [currentTool, setCurrentTool] = useState<Tool>('draw');
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [categories, setCategories] = useState<string[]>([]);

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

  useEffect(() => {
    const loadCategories = async () => {
      try {
        const response = await fetch('/model/data/categories.txt');
        const text = await response.text();
        const categoriesArray = text.split('\n').map(line => line.trim()).filter(line => line);
        setCategories(categoriesArray);
      } catch (error) {
        console.error('Error loading categories:', error);
      }
    };

    loadCategories();
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
      const model = await tf.loadLayersModel('/tfjs_model/model.json');

      const imageData = canvas.toDataURL('image/png');
      setImageDataUrl(imageData);

      const img = new Image();
      img.src = imageData;

      await new Promise((resolve) => {
        img.onload = resolve;
      });

      console.log('Image loaded for classification:', img);

      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

      const predictions = model.predict(tensor) as tf.Tensor;
      const predictionArray = await predictions.array() as number[][];
      const predictedIndex = predictionArray[0].indexOf(Math.max(...predictionArray[0]));

      if (categories.length > 0) {
        const predictedCategory = categories[predictedIndex];
        setPrediction(predictedCategory);
      } else {
        console.error('Categories not loaded');
      }

    } catch (error) {
      console.error('Error using custom model:', error);
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
