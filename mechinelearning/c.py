import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

class LoRAChatBot:
    def __init__(self, base_model_name="Qwen/Qwen2.5-3B", lora_path="./lora_qwen_output"):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        
    def load_model(self):
        """加载LoRA模型"""
        print("正在加载模型...")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.lora_path,
                trust_remote_code=True
            )
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            self.model.eval()  # 设置为评估模式
            
            print("✅ 模型加载完成!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(self, user_input, max_length=200, temperature=0.7):
        """生成回复"""
        try:
            # 构建对话提示
            prompt = self._build_prompt(user_input)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码回复
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取助手回复部分
            if "助手:" in response:
                assistant_response = response.split("助手:")[-1].strip()
            else:
                assistant_response = response.replace(prompt, "").strip()
            
            # 更新对话历史
            self.conversation_history.append({"user": user_input, "assistant": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            return f"生成回复时出错: {str(e)}"
    
    def _build_prompt(self, user_input):
        """构建对话提示"""
        # 简单的单轮对话格式
        return f"用户: {user_input}\n助手: "
    
    def chat_loop(self):
        """启动对话循环"""
        if not self.load_model():
            return
        
        print("\n🤖 LoRA聊天机器人已启动!")
        print("输入 '退出' 或 'quit' 结束对话")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print("再见! 👋")
                    break
                
                if not user_input:
                    print("请输入有效内容")
                    continue
                
                print("🤖 AI: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n再见! 👋")
                break
            except Exception as e:
                print(f"\n❌ 出错: {e}")

# 运行聊天程序
if __name__ == "__main__":
    bot = LoRAChatBot()
    bot.chat_loop()
