export PYTHONPATH="../"

dataset_config="./cfg/MMLongBench.yaml"
# dataset_config="./cfg/m3docVQA.yaml"
# dataset_config="./cfg/Qasper.yaml"

dataset_name=$(basename "$dataset_config" .yaml)


method_name="gbc_standard"

PYTHON_FILE="../Eval/evaluation.py"

python $PYTHON_FILE -d $dataset_config --method $method_name
