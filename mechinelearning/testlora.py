from transformers import TRANSFORMERS_CACHE
print(TRANSFORMERS_CACHE)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import os

class LoRATrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-3B", output_dir="./lora_qwen_output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.lora_config = None

    def setup_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("加载模型和分词器...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        self.model.
        ()

    def create_training_data(self):
        """创建训练数据"""
        sample_data = [
           {"instruction": "介绍人名", "input": "黄婷", "output": "现任中国传媒大学外国语言文化学院葡萄牙语专任讲师、巴西传播研究中心研究员。 她本硕博均毕业于澳门大学，先后获得葡萄牙语文学学士、翻译研究硕士及英语文学博士学位。"},
                       {"instruction": "介绍人名", "input": "张森", "output": "百度高级工程师，现在离职了"},

            {"instruction": "解释人工智能", "input": "", "output": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器和软件。"},
            {"instruction": "写一首关于春天的诗", "input": "", "output": "春风轻拂柳絮飞，\n百花争艳斗芳菲。\n燕子归来寻旧垒，\n人间处处焕生机。"},
            {"instruction": "计算数学表达式", "input": "2 + 3 * 4", "output": "根据数学运算规则，先乘除后加减：3 * 4 = 12，然后 2 + 12 = 14"},
            {"instruction": "翻译成英文", "input": "今天天气很好", "output": "The weather is very good today."},
            {"instruction": "总结文本", "input": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和做出决策。", "output": "机器学习是AI的子集，让计算机无需明确编程即可学习决策。"},
        ]
        return sample_data * 10

    def tokenize_function(self, examples):
        """分词函数"""
        texts = [f"用户: {inst} {inp}\n助手: {out}</s>" for inst, inp, out in 
                zip(examples['instruction'], examples['input'], examples['output'])]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(self):
        """执行训练"""
        print("开始训练...")

        self.setup_model_and_tokenizer()
        self.setup_lora()

        raw_data = self.create_training_data()
        dataset = Dataset.from_list(raw_data)
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

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
            report_to=None,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        print("开始训练循环...")
        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"训练完成！模型保存在: {self.output_dir}")
        return trainer

class ModelTester:
    def __init__(self, base_model_name="Qwen/Qwen2.5-3B", lora_model_path="./lora_qwen_output"):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """加载训练好的模型"""
        print("加载训练好的模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
        self.model.eval()
        
        print("模型加载完成！")
        
    def generate_response(self, instruction, input_text="", max_length=200):
        """生成回答"""
        if input_text:
            prompt = f"用户: {instruction} {input_text}\n助手: "
        else:
            prompt = f"用户: {instruction}\n助手: "
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "助手:" in response:
            assistant_response = response.split("助手:")[-1].strip()
        else:
            assistant_response = response.replace(prompt, "").strip()
            
        return assistant_response
    
    def test_model(self):
        """测试模型表现"""
        test_cases = [
            {"instruction": "解释人工智能", "input": ""},
            {"instruction": "写一首关于夏天的诗", "input": ""},
            {"instruction": "计算数学表达式", "input": "2 + 3 * 4"},
            {"instruction": "翻译成英文", "input": "今天天气很好"},
            {"instruction": "什么是机器学习", "input": ""},
            {"instruction": "介绍", "input": "张森"},
        ]
        
        print("开始测试训练好的模型...")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试 {i}:")
            print(f"问题: {test_case['instruction']} {test_case['input']}")
            
            response = self.generate_response(
                test_case['instruction'], 
                test_case['input']
            )
            
            print(f"回答: {response}")
            print("-" * 50)

def main():
    # 设置镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("=" * 60)
    print("LoRA 微调完整流程")
    print("=" * 60)
    
    # 训练模型
    print("\n1. 开始训练模型...")
    trainer = LoRATrainer()
    trainer.train()
    
    # 测试模型
    print("\n2. 开始测试模型...")
    tester = ModelTester()
    tester.load_model()
    tester.test_model()
    
    print("\n🎉 所有流程完成！")

if __name__ == "__main__":
    main()
