carbot_data_path=carbot_data
rm -rf $carbot_data_path/
cp -r data/dictionary $carbot_data_path
cp -r data/kg $carbot_data_path


python bert_script/run_text_classifier.py \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/predicate_classification \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir $carbot_data_path/model/kbqapc/


python bert_script/run_sequence_labeling.py \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/ner \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir $carbot_data_path/model/ner/