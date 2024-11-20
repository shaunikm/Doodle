import { useRef, useState } from 'react';
import Canvas, { Controls } from './Canvas';

type Tool = 'draw' | 'erase';

function App() {
  const [currentTool, setCurrentTool] = useState<Tool>('draw');
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  return (
    <div className="h-screen overflow-hidden">
      <Canvas
        ref={canvasRef}
        currentTool={currentTool}
        isDrawing={isDrawing}
        setIsDrawing={setIsDrawing}
      />
      <div className="relative w-full p-4 flex justify-between items-center border-b border-[#2a2a2a]/50 bg-[#1a1a1a]/80 backdrop-blur-md z-10">
        <h1 className="text-xl font-medium text-[#efefef]">
          Doodle Placeholder Text
        </h1>
        <div className="flex items-center gap-4">
          <Controls
            currentTool={currentTool}
            setCurrentTool={setCurrentTool}
            onClear={clearCanvas}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
