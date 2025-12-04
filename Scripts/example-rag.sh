sys_config="../config/default.yaml"
dataset_config="./cfg/MMLongBench.yaml"
dataset_name=$(basename "$dataset_config" .yaml)
method_name=$(basename "$sys_config" .yaml)

PYTHON_FILE="../main.py" 

nsplit=2
for num in $(seq 1 $nsplit); do
    log_dir="./logs/${dataset_name}/${method_name}"
    mkdir -p $log_dir
    echo "Log directory created at: $log_dir"

    log_file="${log_dir}/rag-${num}-${method_name}.log"
    echo "Starting RAG for split ${num}..."
    echo "Log file in $log_file"
    
    nohup python $PYTHON_FILE -c $sys_config -d $dataset_config \
        --num $num --nsplit $nsplit rag \
        > $log_file 2>&1 &
done
