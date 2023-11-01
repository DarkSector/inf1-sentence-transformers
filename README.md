# inf1-sentence-transformers
Sentence Transformers on EC2 Inf1 and Amazon SageMaker


## EC2 Inf1
* trace-model.py takes a static batch-size and name of model to build
* inference.py runs the traced model on a single core (Run this 4 times to start 4 different processes)


## Amazon SageMaker
* Build out the static batch-size traced models
* Pull the models on to EFS on SageMaker studio (just upload the model files)
* Update batch size and deploy. 
