# inf1-sentence-transformers
Sentence Transformers on EC2 Inf1 and Amazon SageMaker


## EC2 Inf1
* trace-model.py takes a static batch-size and name of model to build
* inference.py runs the traced model on a single core (Run this 4 times to start 4 different processes)


## Amazon SageMaker
* Build out the static batch-size traced models
* Pull the models on to EFS on SageMaker studio (just upload the model files)
* Update batch size and deploy. 


## Usage

### EC2 - Build
To run on EC2, after installing the relevant packages
You can also change model_id using `--model_id` option
`python trace-model.py --batch_size 50`


### EC2 - Inference
Modify the batch_size and simply run
`python inference.py`
Use Neuron Top (neuron-top) utility on EC2 Inf1 


### SageMaker Inference
Simply upload the model folders and the notebook and run it through SageMaker
