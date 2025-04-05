# Iris Flower Prediction API

A simple project demonstrating how to train a scikit-learn model for Iris flower classification and serve it using a FastAPI web API with a basic HTML interface.

## Project Structure

*   `train_model.py`: Script to train the Random Forest model and save it as `iris_model.pkl`.
*   `main.py`: FastAPI application that loads the model and provides prediction endpoints.
*   `iris_model.pkl`: The pre-trained scikit-learn model file.
*   `templates/`: Directory containing the HTML frontend.
    *   `index.html`: Simple web form to interact with the API.
*   `requirements.txt`: Required Python packages.
*   `.gitignore`: Specifies files/directories ignored by Git.

## Setup

1.  Clone the repository:
    ```bash
    git clone git@github.com:cheezycoding/IrisFlowerPrediction.git
    cd IrisFlowerPrediction
    ```
2.  (Recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running

1.  **Train the model (Optional - pre-trained included):**
    ```bash
    python train_model.py
    ```
    *(This will overwrite `iris_model.pkl`)*
2.  **Run the API server:**
    ```bash
    uvicorn main:app --reload --port 3000
    ```

## Usage

*   **Web Interface:** Once the server is running, open your browser to `http://127.0.0.1:3000`. Use the form to get predictions.
*   **API Endpoint:** Send `POST` requests to `/predict` with JSON data:
    ```bash
    curl -X POST "http://127.0.0.1:3000/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }'
    ```
