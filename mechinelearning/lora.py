import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class LoRATrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", output_dir="./lora_qwen"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.lora_config = None

    def setup_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("加载模型和分词器...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"模型加载完成，参数量: {self.model.num_parameters():,}")

    def setup_lora(self, r=8, lora_alpha=32, lora_dropout=0.1):
        """配置LoRA参数"""
        print("设置LoRA配置...")

        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )

        # 应用LoRA到模型
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # 确保模型在训练模式
        self.model.train()

    def create_sample_data(self):
        """创建示例训练数据"""
        print("创建训练数据...")

        sample_data = [
            {"instruction": "解释人工智能", "input": "",
             "output": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器和软件。"},
            {"instruction": "写一首关于春天的诗", "input": "",
             "output": "春风轻拂柳絮飞，\n百花争艳斗芳菲。\n燕子归来寻旧垒，\n人间处处焕生机。"},
            {"instruction": "计算数学表达式", "input": "2 + 3 * 4",
             "output": "根据数学运算规则，先乘除后加减：3 * 4 = 12，然后 2 + 12 = 14"},
            {"instruction": "翻译成英文", "input": "今天天气很好", "output": "The weather is very good today."},
        ]

        return sample_data * 5

    def format_instruction_data(self, examples):
        """格式化指令数据 - 修复版本"""
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        texts = []
        for i in range(len(instructions)):
            if inputs[i] and inputs[i].strip():
                # 使用更简单的格式，避免特殊token问题
                text = f"用户: {instructions[i]} {inputs[i]}\n助手: {outputs[i]}</s>"
            else:
                text = f"用户: {instructions[i]}\n助手: {outputs[i]}</s>"
            texts.append(text)
        return texts

    def tokenize_function(self, examples):
        """分词函数 - 修复版本"""
        texts = self.format_instruction_data(examples)
        
        print(f"样本文本示例: {texts[0][:100]}...")  # 调试输出
        
        # 分词 - 确保正确设置
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,  # 减少长度
            return_tensors=None,
            add_special_tokens=True,  # 确保添加特殊token
        )
        
        # 关键修复：正确设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    def train(self):
        """执行训练"""
        print("开始训练...")

        # 1. 设置模型和分词器
        self.setup_model_and_tokenizer()

        # 2. 配置LoRA
        self.setup_lora(r=8, lora_alpha=32, lora_dropout=0.1)

        # 3. 准备数据
        raw_data = self.create_sample_data()
        dataset = Dataset.from_list(raw_data)
        
        print(f"数据集样本数: {len(dataset)}")
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # 检查数据格式
        print("检查数据格式...")
        sample = tokenized_dataset[0]
        print(f"input_ids 长度: {len(sample['input_ids'])}")
        print(f"labels 长度: {len(sample['labels'])}")
        print(f"attention_mask 长度: {len(sample['attention_mask'])}")

        # 4. 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            eval_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to=None,
            gradient_checkpointing=False,
        )

        # 5. 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 6. 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # 7. 训练前验证 - 修复版本
        print("训练前验证...")
        self._validate_training_setup(tokenized_dataset)

        # 8. 开始训练
        print("开始训练循环...")
        trainer.train()

        # 9. 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"训练完成！模型保存在: {self.output_dir}")
        return trainer

    def _validate_training_setup(self, tokenized_dataset):
        """验证训练设置是否正确 - 修复版本"""
        print("验证训练设置...")
        
        # 检查模型模式
        print(f"模型训练模式: {self.model.training}")
        
        # 检查是否有可训练参数
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.shape))
        
        print(f"找到 {len(trainable_params)} 个可训练参数")
        
        # 使用真实数据进行测试
        try:
            # 获取一个真实的数据样本
            sample = tokenized_dataset[0]
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(self.model.device)
            labels = torch.tensor(sample['labels']).unsqueeze(0).to(self.model.device)
            
            print(f"测试输入形状: {input_ids.shape}")
            print(f"测试标签形状: {labels.shape}")
            
            # 前向传播
            self.model.train()
            with torch.set_grad_enabled(True):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels  # 关键：传递labels参数
                )
                
                print(f"输出类型: {type(outputs)}")
                print(f"输出属性: {dir(outputs)}")
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                    print(f"✅ 前向传播测试成功, loss: {loss.item():.4f}")
                    
                    # 测试反向传播
                    loss.backward()
                    print("✅ 反向传播测试成功")
                    
                    # 清除梯度
                    self.model.zero_grad()
                else:
                    print("❌ 损失为None，检查数据格式和模型配置")
                    # 尝试其他方式计算损失
                    logits = outputs.logits
                    print(f"Logits形状: {logits.shape}")
                    
                    # 手动计算损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    manual_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    print(f"手动计算损失: {manual_loss.item():.4f}")
                    
        except Exception as e:
            print(f"❌ 训练测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise


# 主执行函数
def main():
    print("开始 LoRA 训练...")
    
    try:
        # 初始化训练器
        trainer = LoRATrainer(
            model_name="Qwen/Qwen2.5-3B",
            output_dir="./lora_qwen_output"
        )

        # 开始训练
        trainer.train()
        print("🎉 训练成功完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
