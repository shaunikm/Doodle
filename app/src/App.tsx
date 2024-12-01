import { useRef, useState, useEffect } from 'react';
import Canvas, { Controls } from './Canvas';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

type Tool = 'draw' | 'erase';

// Define and register the L2 regularizer
class L2 {
  static className = 'L2';

  constructor(config: { l2: number }) {
    return tf.regularizers.l2({ l2: config.l2 });
  }

  static fromConfig(config: {}) {
    return new L2(config as { l2: number });
  }
}

tf.serialization.registerClass(L2 as tf.serialization.SerializableConstructor<L2>);

// Define the Cast layer
class CastLayer extends tf.layers.Layer {
  static className = 'CastLayer';

  constructor() {
    super({});
  }

  call(input: tf.Tensor) {
    return input.cast('float32');
  }

  getConfig() {
    return {};
  }
}

tf.serialization.registerClass(CastLayer);

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
      const model = await tf.loadLayersModel('/model/tfjs_model/model.json');

      const imageData = canvas.toDataURL('image/png');
      setImageDataUrl(imageData);

      const img = new Image();
      img.src = imageData;

      await new Promise((resolve) => {
        img.onload = resolve;
      });

      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims(0);

      const predictions = model.predict(tensor) as tf.Tensor;
      const predictionArray = await predictions.array() as number[][];
      const predictedIndex = predictionArray[0].indexOf(Math.max(...predictionArray[0]));

      if (predictedIndex >= 0 && predictedIndex < categories.length) {
        const predictedCategory = categories[predictedIndex];
        setPrediction(predictedCategory);
      } else {
        console.error('Predicted index out of bounds:', predictedIndex);
      }

    } catch (error) {
      console.error('Error using custom model:', error);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-white overflow-hidden">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#1a1a1a] border-b border-[#2a2a2a]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <h1 className="text-xl font-bold text-white">
              Doodle
            </h1>
            <div className="flex items-center gap-4">
              <Controls
                currentTool={currentTool}
                setCurrentTool={setCurrentTool}
                onClear={clearCanvas}
              />
              <button
                onClick={classifyCanvas}
                className={`p-2.5 rounded-lg transition-all duration-200 border border-[#6366f1] hover:border-[#818cf8] bg-[#6366f1]/10 hover:bg-[#6366f1]/20 group`}
                title="Classify Drawing"
              >
                <MagnifyingGlassIcon className="w-5 h-5 text-[#818cf8] group-hover:text-[#a5b4fc] transition-colors" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Canvas Area */}
      <main className="pt-16 relative h-screen">
        <Canvas
          ref={canvasRef}
          currentTool={currentTool}
          isDrawing={isDrawing}
          setIsDrawing={setIsDrawing}
        />
      </main>

      {/* Prediction Panel */}
      {(imageDataUrl || prediction) && (
        <div className="fixed bottom-0 left-0 right-0 bg-[#1a1a1a] border-t border-[#2a2a2a] transform transition-all duration-300 ease-out">
          <div className="max-w-2xl mx-auto p-6">
            <div className="flex items-start gap-6">
              {imageDataUrl && (
                <div className="flex-1 p-5 rounded-xl bg-[#2a2a2a] border border-[#3a3a3a] shadow-xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-2 h-2 rounded-full bg-[#42b883]" />
                    <p className="text-[#efefef] text-sm font-medium">Captured Drawing</p>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-2xl">
                    <img 
                      src={imageDataUrl} 
                      alt="Captured canvas" 
                      className="max-h-32 w-auto mx-auto object-contain"
                    />
                  </div>
                </div>
              )}
              {prediction && (
                <div className="flex-1 p-5 rounded-xl bg-[#2a2a2a] border border-[#3a3a3a] shadow-xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-2 h-2 rounded-full bg-[#42b883]" />
                    <p className="text-[#efefef] text-sm font-medium">AI Prediction</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-lg bg-[#42b883]/20 flex items-center justify-center">
                      <svg className="w-6 h-6 text-[#42b883]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <p className="text-3xl font-bold text-white">
                      {prediction}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
