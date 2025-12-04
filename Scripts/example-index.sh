index_config="../config/gbc.yaml"

dataset_config="./cfg/Qasper.yaml"

dataset_name=$(basename "$dataset_config" .yaml)
method_name=$(basename "$index_config" .yaml)

PYTHON_FILE="../main.py"

nsplit=2
stage="graph"

for num in $(seq 1 $nsplit); do
    log_dir="./logs/${dataset_name}/index/"
    mkdir -p $log_dir
    echo "Log directory created at: $log_dir"

    log_file="${log_dir}/${method_name}-${num}-index-${stage}.log"
    echo "Starting indexing for split ${num}..."
    echo "Log file in $log_file"
    
    nohup python $PYTHON_FILE -c $index_config -d $dataset_config \
        --num $num --nsplit $nsplit \
        index --stage $stage  \
        > $log_file 2>&1 &
done
