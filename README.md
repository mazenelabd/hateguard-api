# HateGuard API
This is the second layer of a three-layer product:
1. [Machine learning layer](https://github.com/mazenelabd/hateguard-machine-learning) or Data science layer.
2. Back-end layer, the current repository.
3. [Front-end layer](https://github.com/mazenelabd/HateGuard/).

To run the product code, you do not have to rebuild the model in the first layer you can start from this layer.

This layer is a Flask API that utilises the quantized ONNX model from the [Machine learning layer](https://github.com/mazenelabd/hateguard-machine-learning).
To start the API, you should open the Command Prompt and make sure that you have Python installed on your machine, then go to the project directory and follow the following commands to run the API.

1- Open the Command Prompt.

2- Make sure that you have Python installed on your machine
```
python -V
```
3- Go to the project directory from the command prompt
```
cd where-the-project-is-located-on-your-machine
```
4- Create a virtual environment
```
python -m venv venv
```
5- Activate the virtual environment
.\venv\Scripts\activate

6- Install pip for the virtual environment
```
python -m pip install --upgrade pip
```
7- Install the required packages
```
python -m pip install -r requirements.txt
```
8- Create ".env" file that includes:
```
SECRET_KEY="YOUR SECRET KEY"
model_name="microsoft/MiniLM-L12-H384-uncased"
onnx_model_name="model_int8.onnx"
```
9- Run the API
```
python -m flask --app ./app.py run
```
10- In your browser go to:
http://127.0.0.1:5000
you should see a page saying: "Welcome to the HateGuard API By Mazen Elabd"

The next step is to run the [frontend](https://github.com/mazenelabd/HateGuard/)
