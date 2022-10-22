# How to Use Pretrained Language Model #

### Download Pretrained Model ###
You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M)(config.json, merges.txt, pytorch_model.bin, and vocab.json) model insides ./models/. 
Here, we use BlenderBot as the example, and more models are provided by hugging face.

### Data and Data Processing ###
First, we need the dataset files insides ./data/. Then, run prepare.py.

### Finetune the model ###
```console
❱❱❱ python train.py --eval_input_file ./data/valid.txt --seed 13 --max_input_length 160 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --num_epochs 2 --warmup_steps 100 --loss_scale 0.0
```

### Interact with the User ###
The basic use of the trained model is given. But the detailed interaction should be finished for the final results.
