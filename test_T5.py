import transformers
from transformers.pipelines.text2text_generation import ReturnType, Text2TextGenerationPipeline

model_id = 't5-3b'

model = transformers.T5ForConditionalGeneration.from_pretrained(model_id)

print(model.config.task_specific_params)



# CUDA_VISIBLE_DEVICES=0 python test_T5.py