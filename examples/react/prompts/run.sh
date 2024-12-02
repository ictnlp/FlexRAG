#!/bin/bash

LIB_PATH="[path_to_librarian]"
TEST_PATH="[path_to_testset]"
MODEL_NAME="[model_name]"

python $LIB_PATH/scripts/run_assistant.py \
    user_module=$LIB_PATH/examples/react \
    data_path=$TEST_PATH \
    assistant_type=react \
    "react_config.generator_type=openai \
    react_config.openai_config.model_name=$MODEL_NAME \
    react_config.do_sample=False
    retrieval_eval_config.retrieval_metrics=[success_rate] \
    retrieval_eval_config.evaluate_field=summary \
    response_eval_config.answer_preprocess_pipeline.processor_type=[simplify_answer] \
    response_eval_config.response_metrics=[f1,recall,em,precision,accuracy]
