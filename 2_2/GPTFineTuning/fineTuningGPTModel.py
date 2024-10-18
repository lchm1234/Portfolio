import torch
import transformers
import peft
import datasets
import trl
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

print(f'Torch version : {torch.__version__}')
print(f'Transformers version : {transformers.__version__}')
print(f'Peft version : {peft.__version__}')
print(f'Datasets version : {datasets.__version__}')
print(f'Trl version : {trl.__version__}')
    
if torch.cuda.is_available():
    # Print CUDA version
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available on this system.")
    
base_model, new_model = 'maywell/Mistral-ko-7B-v0.1', 'zenGPT/zenGPT_mistral_7b'

dataset = load_dataset('csv', data_files='./fineTuningData.csv', encoding='cp949')
dataset = dataset['train']

# 데이터셋 크기 확인
dataset_size = len(dataset)
print("Dataset size:", dataset_size)

# 데이터셋 분할
random.seed(42)
indices = list(range(dataset_size))
random.shuffle(indices)

split_idx = int(0.9 * dataset_size)
train_indices, test_indices = indices[:split_idx], indices[split_idx:]

train_dataset = dataset.select(train_indices)
test_dataset = dataset.select(test_indices)

print(train_dataset['chat_sample'][0])
print(train_dataset.shape)

# 인덱스 범위 조절
num_selected_samples = min(1000, len(train_dataset))
train_dataset = train_dataset.shuffle(seed=42).select(range(num_selected_samples))

# 베이스 모델 불러오기
bits_and_bytes_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bits_and_bytes_config,
    device_map={"":0},
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

# 하이퍼파라미터
training_arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim='paged_adamw_8bit',
    save_steps=5000,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type='constant'
)

# sft 파라미터
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field='chat_sample',
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False
)

trainer.train()
# save fine tuning model
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
model.config.use_cache = True
model.eval