import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import os

class LoRAComparison:
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B", lora_model_path="./lora_qwen_output"):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.base_tokenizer = None
        self.base_model = None
        self.lora_tokenizer = None
        self.lora_model = None
        
    def load_models(self):
        """加载基础模型和LoRA模型"""
        print("正在加载模型...")
        
        try:
            # 加载基础模型（训练前）
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ 基础模型加载完成")
            
            # 加载LoRA模型（训练后）
            self.lora_tokenizer = AutoTokenizer.from_pretrained(
                self.lora_model_path,
                trust_remote_code=True
            )
            
            base_model_for_lora = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.lora_model = PeftModel.from_pretrained(base_model_for_lora, self.lora_model_path)
            print("✅ LoRA模型加载完成")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(self, model, tokenizer, prompt, max_length=200):
        """生成模型回答"""
        try:
            # 使用训练时的相同格式
            if "输入:" in prompt:
                instruction, input_text = prompt.split("输入:")
                formatted_prompt = f"用户: {instruction.strip()}\n{input_text.strip()}\n助手: "
            else:
                formatted_prompt = f"用户: {prompt}\n助手: "
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取助手回答部分
            if "助手:" in response:
                assistant_response = response.split("助手:")[-1].strip()
            else:
                assistant_response = response.replace(formatted_prompt, "").strip()
                
            return assistant_response
            
        except Exception as e:
            return f"生成失败: {str(e)}"
    
    def compare_responses(self, test_cases):
        """对比基础模型和LoRA模型的回答"""
        if not self.load_models():
            return None
            
        print("开始模型对比测试...")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"测试 {i}/{len(test_cases)}: {test_case}")
            
            # 基础模型回答
            base_response = self.generate_response(
                self.base_model, self.base_tokenizer, test_case
            )
            
            # LoRA模型回答
            lora_response = self.generate_response(
                self.lora_model, self.lora_tokenizer, test_case
            )
            
            results.append({
                'question': test_case,
                'base_model': base_response,
                'lora_model': lora_response,
                'base_length': len(base_response),
                'lora_length': len(lora_response),
                'improvement': len(lora_response) - len(base_response)
            })
            
            print(f"  - 基础模型: {base_response[:50]}...")
            print(f"  - LoRA模型: {lora_response[:50]}...")
            print()
        
        return results
    
    def save_comparison_report(self, results, output_file="lora_comparison_report.txt"):
        """保存对比报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LoRA微调前后对比报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"基础模型: {self.base_model_name}\n")
            f.write(f"LoRA模型: {self.lora_model_path}\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"测试案例 {i}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"问题: {result['question']}\n\n")
                
                f.write("基础模型回答:\n")
                f.write(f"{result['base_model']}\n\n")
                
                f.write("LoRA模型回答:\n")
                f.write(f"{result['lora_model']}\n\n")
                
                f.write("分析:\n")
                f.write(f"- 基础模型回答长度: {result['base_length']} 字符\n")
                f.write(f"- LoRA模型回答长度: {result['lora_length']} 字符\n")
                f.write(f"- 长度变化: {result['improvement']:+d} 字符\n")
                
                # 简单的质量评估
                if result['improvement'] > 10:
                    f.write("- 评估: 回答更加详细丰富\n")
                elif result['improvement'] < -5:
                    f.write("- 评估: 回答更加简洁\n")
                else:
                    f.write("- 评估: 变化不明显\n")
                    
                f.write("=" * 80 + "\n\n")
        
        print(f"✅ 对比报告已保存: {output_file}")
    
    def create_comparison_csv(self, results, output_file="lora_comparison.csv"):
        """创建CSV格式的对比数据"""
        df_data = []
        for result in results:
            df_data.append({
                'question': result['question'],
                'base_model_response': result['base_model'],
                'lora_model_response': result['lora_model'],
                'base_length': result['base_length'],
                'lora_length': result['lora_length'],
                'length_difference': result['improvement']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ CSV数据已保存: {output_file}")
        
        # 显示统计信息
        print("\n统计信息:")
        print(f"平均回答长度 - 基础模型: {df['base_length'].mean():.1f} 字符")
        print(f"平均回答长度 - LoRA模型: {df['lora_length'].mean():.1f} 字符")
        print(f"平均长度变化: {df['length_difference'].mean():.1f} 字符")
        
        return df
    
    def analyze_improvements(self, results):
        """分析改进情况"""
        print("\n改进分析:")
        print("-" * 40)
        
        total_improvement = 0
        improved_cases = 0
        
        for i, result in enumerate(results, 1):
            improvement = result['improvement']
            total_improvement += improvement
            
            if improvement > 0:
                improved_cases += 1
                status = "✓ 改进"
            elif improvement < 0:
                status = "⚠️ 变差"
            else:
                status = "➖ 无变化"
                
            print(f"案例 {i}: {improvement:+d} 字符 - {status}")
        
        print("-" * 40)
        print(f"平均改进: {total_improvement/len(results):.1f} 字符")
        print(f"改进案例: {improved_cases}/{len(results)}")
        print(f"改进率: {improved_cases/len(results)*100:.1f}%")

def main():
    # 测试案例 - 使用您训练时类似的问题
    test_cases = [
        "解释人工智能",
        "写一首关于春天的诗",
        "计算数学表达式 2+3*4",
        "什么是机器学习",
        "翻译成英文: 今天天气很好",
        "总结文本: 深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的抽象表示",
        "写一个简单的Python函数计算斐波那契数列"
    ]
    
    print("LoRA微调效果对比测试")
    print("=" * 60)
    
    # 创建对比实例
    comparator = LoRAComparison(
        base_model_name="Qwen/Qwen2.5-0.5B",
        lora_model_path="./lora_qwen_output"  # 您训练保存的路径
    )
    
    # 执行对比
    results = comparator.compare_responses(test_cases)
    
    if results:
        # 保存报告
        comparator.save_comparison_report(results)
        
        # 创建CSV数据
        df = comparator.create_comparison_csv(results)
        
        # 分析改进
        comparator.analyze_improvements(results)
        
        print("\n🎉 对比完成!")
        print("生成的文件:")
        print("  📝 lora_comparison_report.txt - 详细对比报告")
        print("  📊 lora_comparison.csv - 数据表格")
        
    else:
        print("❌ 对比测试失败")

if __name__ == "__main__":
    main()
