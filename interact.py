from BlenderbotModel import ChatModel

model_config = {'use_gpu': False,
                'pretrained_model_path': "models/Blenderbot_small-90M",
                'max_input_length': 256,
                'max_src_turn': 4,
                'max_decode_length': 64,
                'min_decode_length': 1,
                'temperature': 0.8,
                'top_k': 0,
                'top_p': 0.9,
                'num_beams': 1,
                'repetition_penalty': 1.0,
                'no_repeat_ngram_size': 3,
                'checkpoint': './DATA/2022-10-22141842.3e-05.16.1gpu/epoch-1.bin', # you need to replace this with the checkpoint path.
                }


# this part can be changed according to your design.
chatbot = ChatModel(**model_config)
dialog_context = {"dialog": [{"text": "Hello, nice to meet you!", "speaker": "sys"}]}
response = chatbot.interact(dialog_context)
print(response)
