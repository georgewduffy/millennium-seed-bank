## Predicting seed health in partnership with the UK's 'doomsday vault'

A project with Kew Gardens and the Millennium Seed Bank to provide a web application for predicting and allocating seed health scores. Below is a map of where the core functionality is located within the directory:
* Frontend: `web/`
* Server: `flask_server/`
* Synthesiser (creates synthetic image data/seed x-rays): `src/synth/`
* Model training: `notebooks/yolov8_training.ipynb` and `src/mask_rcnn_training`

*Note this repo has been forked from my Imperial College London research account to my personal GitHub so the CI pipeline and downloading trained model weights won't work because of the GitLab config and permissioned access to Imperial's bitbucket.*

### Requirements
1. Python >= 3.10 - [Download](https://www.python.org/downloads/)
2. Node.js >= 20.0 - [Download](https://nodejs.org/en/download)

### Setup
1. Clone the repository
2. Create a python virtual environment in the project root directory
    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. Download required node packages
    ```
    cd web/
    npm install
    ```
4. Download `yolo.pt` and `rcnn.pt` model weights. From the root directory
    ```
    cd flask_server/app/models/final_model_weights/
    wget https://www.doc.ic.ac.uk/project/2023/70079/g237007903/weights/rcnn.pt
    wget https://www.doc.ic.ac.uk/project/2023/70079/g237007903/weights/yolo.pt
    ```

### Running the application
1. Start the back-end. From the root directory
    ```
    source venv/bin/activate
    cd flask_server/
    flask run
    ```
2. In a new terminal, start the front-end. From the root directory
    ```
    cd web/
    npm start
    ```

Your default web broswer should open a new tab/window with the web application served at [http://localhost:3000](http://localhost:3000)

## Running tests
From the root directory
```
source venv/bin/activate
pytest
```
