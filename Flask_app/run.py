from app import create_app
# Import function that builds and configures the Flask app

from app.ml import load_model
# Import function that loads the YOLO model into memory

# Create Flask app instance by calling factory function
app = create_app()

# This block runs only when this file is executed directly (python run.py)
if __name__ == "__main__":
    
    # Load YOLO model once at startup (so first request is faster)
    load_model()
    
    # Start Flask development server
    # debug=True → auto reload when code changes (development only)
    # host="127.0.0.1" → runs on local machine
    # port=5000 → default Flask port
    app.run(debug=True, host="127.0.0.1", port=5000)