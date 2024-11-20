import React, { forwardRef, useEffect, useRef, useState } from 'react';
import { PencilIcon, XMarkIcon as EraserIcon, TrashIcon } from '@heroicons/react/24/outline';

type Tool = 'draw' | 'erase';

interface ControlsProps {
  currentTool: Tool;
  setCurrentTool: (tool: Tool) => void;
  onClear: () => void;
}

export const Controls: React.FC<ControlsProps> = ({ currentTool, setCurrentTool, onClear }) => {
  return (
    <div className="flex gap-3">
      <button
        onClick={() => setCurrentTool('draw')}
        className={`p-2.5 rounded transition-all duration-200 border border-[#2a2a2a] hover:border-[#42b883]/50 group ${
          currentTool === 'draw'
            ? 'bg-[#42b883]/20 border-[#42b883] shadow-[0_0_12px_0_rgba(66,184,131,0.3)]'
            : 'bg-[#2a2a2a] hover:bg-[#333]'
        }`}
        title="Draw"
      >
        <PencilIcon className={`w-5 h-5 transition-colors ${
          currentTool === 'draw' 
            ? 'text-[#42b883]' 
            : 'text-[#efefef] group-hover:text-[#42b883]'
        }`} />
      </button>
      <button
        onClick={() => setCurrentTool('erase')}
        className={`p-2.5 rounded transition-all duration-200 border border-[#2a2a2a] hover:border-[#94a3b8]/50 group ${
          currentTool === 'erase'
            ? 'bg-[#94a3b8]/20 border-[#94a3b8] shadow-[0_0_12px_0_rgba(148,163,184,0.3)]'
            : 'bg-[#2a2a2a] hover:bg-[#333]'
        }`}
        title="Erase"
      >
        <EraserIcon className={`w-5 h-5 transition-colors ${
          currentTool === 'erase' 
            ? 'text-[#94a3b8]' 
            : 'text-[#efefef] group-hover:text-[#94a3b8]'
        }`} />
      </button>
      <button
        onClick={onClear}
        className="p-2.5 rounded transition-all duration-200 border border-[#2a2a2a] hover:border-[#ff6b6b]/50 hover:bg-[#333] bg-[#2a2a2a] group"
        title="Clear"
      >
        <TrashIcon className="w-5 h-5 text-[#ff6b6b] opacity-80 group-hover:opacity-100 transition-opacity" />
      </button>
    </div>
  );
};

interface CanvasProps {
  currentTool: Tool;
  isDrawing: boolean;
  setIsDrawing: (isDrawing: boolean) => void;
}

interface CursorProps {
  position: { x: number; y: number };
  tool: Tool;
}

const Cursor: React.FC<CursorProps> = ({ position, tool }) => {
  return (
    <div 
      className={`pointer-events-none fixed rounded-full border transform -translate-x-1/2 -translate-y-1/2 transition-colors duration-150 ${
        tool === 'draw' 
          ? 'border-[#42b883] w-2 h-2 border' 
          : 'border-[#94a3b8] w-5 h-5 border'
      }`}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    />
  );
};

const Canvas = forwardRef<HTMLCanvasElement, CanvasProps>(
  ({ currentTool, isDrawing, setIsDrawing }, ref) => {
    const lastPoint = useRef<{ x: number; y: number } | null>(null);
    const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });
    const [showCursor, setShowCursor] = useState(false);

    useEffect(() => {
      const canvas = ref as React.MutableRefObject<HTMLCanvasElement>;
      if (!canvas.current) return;

      const updateCanvasSize = () => {
        canvas.current.width = window.innerWidth;
        canvas.current.height = window.innerHeight;
      };

      updateCanvasSize();
      window.addEventListener('resize', updateCanvasSize);

      return () => window.removeEventListener('resize', updateCanvasSize);
    }, []);

    const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
      return {
        x: e.clientX,
        y: e.clientY
      };
    };

    const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const point = getCanvasPoint(e);
      lastPoint.current = point;
      setIsDrawing(true);

      const canvas = ref as React.MutableRefObject<HTMLCanvasElement>;
      const ctx = canvas.current.getContext('2d');
      if (!ctx) return;

      ctx.beginPath();
      ctx.arc(point.x, point.y, currentTool === 'draw' ? 3 : 10, 0, Math.PI * 2);
      ctx.fillStyle = currentTool === 'draw' ? '#000' : '#fff';
      ctx.fill();
    };

    const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing || !lastPoint.current) return;

      const canvas = ref as React.MutableRefObject<HTMLCanvasElement>;
      const ctx = canvas.current.getContext('2d');
      if (!ctx) return;

      const point = getCanvasPoint(e);

      ctx.beginPath();
      ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
      ctx.lineTo(point.x, point.y);
      
      ctx.strokeStyle = currentTool === 'draw' ? '#000' : '#fff';
      ctx.lineWidth = currentTool === 'draw' ? 6 : 20;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.stroke();
      lastPoint.current = point;
    };

    const stopDrawing = () => {
      lastPoint.current = null;
      setIsDrawing(false);
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
      setCursorPosition({
        x: e.clientX,
        y: e.clientY
      });
    };

    const handleMouseEnter = () => {
      setShowCursor(true);
    };

    const handleMouseLeave = () => {
      setShowCursor(false);
      stopDrawing();
    };

    return (
      <>
        <canvas
          ref={ref}
          className="fixed inset-0 bg-white"
          onMouseDown={startDrawing}
          onMouseMove={(e) => {
            draw(e);
            handleMouseMove(e);
          }}
          onMouseUp={stopDrawing}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        />
        {showCursor && (
          <Cursor 
            position={cursorPosition} 
            tool={currentTool} 
          />
        )}
      </>
    );
  }
);

Canvas.displayName = 'Canvas';

export default Canvas;