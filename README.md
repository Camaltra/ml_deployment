# ML GitOps Pipeline

## Table of Contents
1. [Overview](#overview)
   - [Summary](#summary)
   - [Goal](#Goal)
   - [Constraints](#contraints)
   
2. [Installation](#installation)

3. [Usage](#usage)
   - [Experiments](#experiments)
   - [API](#api)
4. [Output and model description](#output-and-model-description)
5. [Acknowledgements](#acknowledgements)
6. [License](#license)
7. [Contributing](#contributing)

## Overview

### Summary

The project use Iterative.ai tool ([DVC](https://dvc.org) & [CML](https://cml.dev) (Implementation CML soon)) to train a model and keep track of each experiments. Its allow to visualize them and stock them (Data and Model) into a S3 repo that serve as **Data and Model registery**.  
It all aim with the goal of creating reproducable experiments and, by leveraging metrics, push them into the dev environement.   
On the last process, while pushing on main (PR), a pipeline will take care of building a docker image of the app and push it to DockerHub.   
This all end with and end-to-end ML pipeline to experiments and ship model to prod

### Goal

The project is build on this machine learning model repo R&D [Areal Road Seg](https://github.com/Camaltra/aerial_road_segementation) which leverage a U-Net to create a segmentation of road view from the sky.   
The dataset used is the [Ottawa Road Dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjE9ZW-_oKEAxUJAfsDHfyTB2cQFnoECA4QAQ&url=https%3A%2F%2Fgithub.com%2Fyhlleo%2FRoadNet&usg=AOvVaw0tbX4fQPi7WUIfvIPr4glA&opi=89978449)    
Find more information about the model, metrics, loss, optim and so on in the R&D repo.


#### Constraints

Only use the GitOps tool, no use of third part software to vizualize experiments and keep models artifact.

## Installation

To use this codebase, follow these steps:
By Preference, please fork it to make pipeline work for you while deploying model. Then clone it
```bash
git clone https://github.com/.../ml_deployment.git
cd ml_deployment
```
[Install DVC](https://dvc.org/doc/install)


Create an S3 bucket with IAM user in AWS (or use other Cloud plateform (You may need to change the requirements.txt)) and connect your account using AWS CLI. Change the DVC config file `.dvc/config` to connect to your storage.
Install the DVC requirements
```bash
pytho3n -m venv venv
source venv/bin/activate
pip install -r dvc.requirements.txt
```

Install the dvc version of your Cloud Provider ie `pip install 'dvc['s3']'`   

To get a better experience, please install the DVC extension on VSCode (Other Editor do not have this extension)

Make sure Docker is installed.    

After fork the REPO, get into the secret section and fill `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DOCKER_HUB_USERNAME`, `DOCKER_HUB_ACCESS_TOKEN`

You should also change the out Docker Hub repo in the CI/CD to push to your own one

The you are good to go, and can follow the next step to run experiements.

## Usage

### Experiments
Checkout to experiments branch. Modify the `params.yaml` to change the model training and configurations. Then, when you satified with the parameters, go to the DVC window and click on `Run Experiements`.   
You will get output file, as well as the model and metrics. If you conviced that this experiements is good to share, push it into and new banch (Icon on the experiment tab)   
Then you can create PR using the image in the `src/training` images as well as the metrics present in the evaluation json file.   
After merging the PR into experiments, run `dvc repro` to run the experiement again and then push the data and the model to S3 using `dvc push`. The dvc.lock file will keep track of the data and the model.

**Update Soon**
A CI/CD pipeline will be implemented to train the model after pushing to the branch, push the content to DVC and create a PR direclty.   
A CI/CD Pipeline will be implemented to clean non merged experiments's data on S3   
So that with these update, you will jsut need to try and play with experiments and the deployment part will be abstract.

### API
For the API part, after merging your experiements branch to dev, and then to main, a CI/CD pipeline will build docker image with the model tracked by dvc.lock file. Then it will push it to the hub.   
To make it work, pull it from the Hub, and just run
```
docker run -pOUT_PORT:5000 IMAGE_NAME
```
You can now it the server and get prediction (One image by One) through `http://localhost:OUT_PORT/predict`

## Acknowledgements

This project draws inspiration from the differents WorkShop of Iterative.ai and the available documentation through they website.

## License

This project is licensed under the [Apache 2.0] - see the LICENSE.md file for details.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. We welcome any feedback, suggestions, or improvements.
