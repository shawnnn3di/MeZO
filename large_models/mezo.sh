MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-8}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-10000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SCOPE=${SCOPE:-1}
SAMPLE_SCHEME=${SAMPLE_SCHEME:-default}
WINDOW_WIDTH=${WINDOW_WIDTH:-0}
MU=${MU:-0.9}
ZOO_NAME=${ZOO_NAME:-default}
NESTEROV=${NESTEROV:-False}
STOP_MOMENTUM=${STOP_MOMENTUM:-5000}


MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

NESTEROV_ARGS=""
if [ "$NESTEROV" == "True" ]; then 
    NESTEROV_ARGS="--nesterov True"
fi

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$SCOPE-$SAMPLE_SCHEME-mu$MU-ww$WINDOW_WIDTH-sm$STOP_MOMENTUM-zoo_$ZOO_NAME

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --scope $SCOPE \
    --sample_scheme $SAMPLE_SCHEME \
    --window_width $WINDOW_WIDTH \
    --mu $MU \
    --zoo_name $ZOO_NAME \
    --stop_momentum $STOP_MOMENTUM \
    $EXTRA_ARGS \
    $TASK_ARGS \
    $NESTEROV_ARGS \
    "$@"
