# SynAnno

## How to use the tool
### Locally
##### Requirements
- Python 3 (https://www.python.org/downloads/)
- Hardware: 1-8 Nvidia GPUs with at least 12G GPU memory (change SYSTEM.NUM_GPU accordingly based on the configuration of your machine)
##### Instructions
1. Clone this repository to your system.</li>
2. Navigate to the cloned repository using your terminal. </br>
``` cd  "cloned repository path goes here" ```
3. Install the required dependencies. </br>
```pip install -r requirements.txt```
4. Launch the app and start annotating!</br>
```python -m flask run``` 
5. Access </br>
```localhost:8080``` 

### With Docker
##### Requirements
- Docker (https://docs.docker.com/get-docker/)
##### Instructions 
1. Clone this repository to your system.</li>
2. Navigate to the cloned repository using your terminal. </br>
``` cd  "cloned repository path goes here" ```
3. Create the docker image. </br>
``` docker build --tag "name" .```
4. Run the docker image. </br>
``` docker run --publish 8080:8080 "name"```
5. Access </br>
```localhost:8080```

### Additional instructions
1. If starting a new proofreading task, upload the .h5 files, proofread and remember to save the results at the end. It is going to take some minutes to load the .h5 files, do not worry. 
2. If continuing a previously incomplete proofreading task or just going over it, no need to upload the .h5 files again. Just upload the JSON file (saved results) from the previous session and you shall be ready!
