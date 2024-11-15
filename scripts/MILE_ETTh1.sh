export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_id_name=ETTh1
model_name=MILE
data=ETTh1_MI
root_path_name=../all_datasets/ETT-small/
bs=128
lr=0.005

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'96 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size 128 \
  --learning_rate 0.005 \
  --seq_len 96 \
  --pred_len 96 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'192 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 192 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'336 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 336 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'720 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 720 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'960 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 960 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'1080 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 1080 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'1200 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 1200 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'1440 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 1440 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'1560 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 1560 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path ETTh1.csv \
  --model_id $model_id_name'_'96'_'1680 \
  --model $model_name \
  --data $data \
  --features M \
  --batch_size $bs \
  --learning_rate $lr \
  --seq_len 96 \
  --pred_len 1680 \
  --window_len 96 192 384 \
  --train_epochs 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
  #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len''$window_len.log