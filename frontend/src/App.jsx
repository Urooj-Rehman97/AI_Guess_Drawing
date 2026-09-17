import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  Brain,
  Sparkles,
  RotateCcw,
  WandSparkles,
  Trophy,
} from "lucide-react";

import "./App.css";

const categories = [
  "star",
  "moon",
  "apple",
  "ambulance",
  "basket",
  "banana",
  "bat",
  "bee",
  "book",
];

function App() {
  const canvasRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [topPredictions, setTopPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const rect = canvas.getBoundingClientRect();

    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

    ctx.beginPath();
    ctx.moveTo(x, y);

    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const rect = canvas.getBoundingClientRect();

    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

    ctx.lineTo(x, y);

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#050505";

    ctx.fillRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    setPrediction(null);
    setConfidence(null);
    setTopPredictions([]);
    setLoading(false);
  };

  const guessDrawing = async () => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    // AI thinking state
    setPrediction("AI THINKING...");
    setConfidence(null);
    setTopPredictions([]);
    setLoading(true);

    try {
      // Canvas ko PNG Blob mein convert karo
      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/png");
      });

      // FormData create karo
      const formData = new FormData();

      // Flask ke request.files["image"] ke saath match
      formData.append(
        "image",
        blob,
        "drawing.png"
      );

      // Flask API ko image send karo
      const response = await axios.post(
        "https://ai-guess-drawing.onrender.com/predict",
        formData
      );

      console.log(
        "Flask Response:",
        response.data
      );

      // ==========================
      // MAIN PREDICTION
      // ==========================

      setPrediction(
        response.data.prediction.toUpperCase()
      );

      // ==========================
      // CONFIDENCE
      // ==========================

      setConfidence(
        `${response.data.confidence.toFixed(2)}%`
      );

      // ==========================
      // TOP 3 PREDICTIONS
      // ==========================

      setTopPredictions(
        response.data.top_predictions || []
      );

    } catch (error) {

      console.error(
        "Prediction Error:",
        error
      );

      // Flask ka error response
      if (error.response) {

        console.error(
          "Flask Error:",
          error.response.data
        );

      }

      // Network/server error
      else if (error.request) {

        console.error(
          "Server Error: Flask server response nahi de raha."
        );

      }

      // Other error
      else {

        console.error(
          "Request Error:",
          error.message
        );

      }

      setPrediction("ERROR");
      setConfidence(null);
      setTopPredictions([]);

    } finally {

      setLoading(false);

    }
  };
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);
  return (
    <div className="app">

      {/* Navbar */}

      <nav className="navbar">

        <div className="logo">
          <div className="logo-icon">
            <Brain size={25} />
          </div>

          <div>
            <h2>Guess My Drawing</h2>
            <span>AI Vision</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI ONLINE
        </div>

      </nav>


      {/* Main */}

      <main className="main">

        {/* Heading */}

        <section className="hero">

          <div className="badge">
            <Sparkles size={16} />
            AI POWERED
          </div>

          <h1>
            Draw it.
            <span> Let AI guess it.</span>
          </h1>

          <p>
            Draw anything inside the canvas and our
            AI model will try to recognize it.
          </p>

        </section>


        {/* Workspace */}

        <section className="workspace">


          {/* Drawing Card */}

          <div className="drawing-card">

            <div className="card-header">

              <div>
                <h3>Your Drawing</h3>

                <p>
                  Use your mouse to draw
                </p>
              </div>

              <WandSparkles size={22} />

            </div>


            <div className="canvas-wrapper">

              <canvas
                ref={canvasRef}
                width={500}
                height={500}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
              />

              <div className="canvas-hint">
                Start drawing here...
              </div>

            </div>


            <div className="buttons">

              <button
                className="clear-btn"
                onClick={clearCanvas}
              >
                <RotateCcw size={18} />
                Clear
              </button>

              <button
                className="guess-btn"
                onClick={guessDrawing}
              >
                <Brain size={18} />
                Guess Drawing
              </button>

            </div>

          </div>


          {/* Result Card */}

          <div className="result-card">

            <div className="result-header">

              <div className="result-icon">
                <Brain size={25} />
              </div>

              <div>
                <h3>AI Prediction</h3>

                <p>
                  Model analysis
                </p>
              </div>

            </div>


            {!prediction && (

              <div className="empty-result">

                <div className="empty-icon">
                  <Sparkles size={30} />
                </div>

                <h3>Waiting for your drawing</h3>

                <p>
                  Draw something and click
                  <b> Guess Drawing </b>
                  to see what AI thinks.
                </p>

              </div>

            )}


            {prediction && (

              <div className="prediction">

                <div className="thinking">
                  🤖 AI THINKS...
                </div>

                <div className="prediction-name">
                  {prediction}
                </div>

                {confidence && (

                  <>

                    <div className="confidence-title">
                      Confidence
                    </div>

                    <div className="confidence-value">
                      {confidence}
                    </div>

                    <div className="progress">

                      <div
                        className="progress-bar"
                        style={{
                          width: confidence
                        }}
                      ></div>

                    </div>


                    <div className="top-title">
                      <Trophy size={17} />
                      Top Predictions
                    </div>

                    <div className="top-list">

                      {topPredictions.map((item, index) => (

                        <div key={index}>

                          <span>{index + 1}</span>

                          {item.label.toUpperCase()}

                          <b>
                            {item.confidence.toFixed(2)}%
                          </b>

                        </div>

                      ))}

                    </div>
                  </>

                )}

              </div>

            )}

          </div>

        </section>


        {/* Categories */}

        <section className="categories">

          <p>AI CAN RECOGNIZE</p>

          <div>

            {categories.map((category) => (

              <span key={category}>
                {category}
              </span>

            ))}

          </div>

        </section>

      </main>


      <footer>
        Built with React + Flask + CNN
      </footer>

    </div>
  );
}

export default App;