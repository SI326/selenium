from flask import Flask, render_template, jsonify
import os
import analysis

app = Flask(__name__)

# Global variable to store the latest result temporarily (in-memory)
LATEST_RESULT = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global LATEST_RESULT
    try:
        # 1. Capture Image
        # static/screenshots is relative to where app.py is run
        filename, filepath = analysis.capture_mosdac_image(output_dir="static/screenshots")
        
        # 2. Analyze Image
        # We pass "static" as static_dir so heatmaps go to static/heatmaps
        result = analysis.analyze_tcc(filename, filepath, static_dir="static")
        
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 400
            
        LATEST_RESULT = result
        return jsonify({"status": "success", "data": result})
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(LATEST_RESULT)

if __name__ == '__main__':
    # Ensure static directories exist
    os.makedirs("static/screenshots", exist_ok=True)
    os.makedirs("static/heatmaps", exist_ok=True)
    app.run(debug=True, port=5001)
