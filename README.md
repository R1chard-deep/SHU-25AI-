# SHU-25AI-YQS-'LLM Fine-tuning'
**基于 LoRA 对 Gemma 2B 进行风格微调（Leprechaun & Yoda 风格）**

本项目演示了如何使用 **LoRA** 对大型语言模型 **Gemma 2B（google/gemma-2-2b-it）** 进行风格迁移微调。通过对标准英文回答进行再训练，使模型能够以两种特殊风格回答问题：

1. Leprechaun 风格 —— 轻快、戏谑、带有爱尔兰民间传说口吻
2. Yoda 风格 —— 语序倒置、哲理化表达方式（星战中尤达大师的说话方式）

本项目主要内容包括：
- LLM 聊天模板输入与输出构建
- 使用 LoRA 的参数高效微调
- 自定义对话生成函数
- 基于更大模型的“LLM 判官”风格一致性评估
- 对比得分形成定量化风格指标


## 1. 模型与数据集

基础模型：
google/gemma-2-2b-it

使用的数据集：
| 风格 | 内容描述 | 用途 |
|------|---------|------|
| Base Q/A | 标准英文问答 | 基线模型行为参考 |
| Leprechaun Q/A | 人工生成的妖精语气回答 | 第一阶段微调 |
| Yoda Q/A | 模仿 Yoda 语序风格的回答 | 第二阶段微调与最终评估目标 |


## 2. 方法说明

### 2.1 对话模板（Chat Prompt）
<start_of_turn>user  
{question}  
<end_of_turn>  
<start_of_turn>model

### 2.2 LoRA 微调策略
仅微调模型中约 0.4% 参数，保持模型高效训练与推理。

LoRA 应用于：
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
LoRA rank = 8

### 2.3 训练流程概述
1) 加载基础模型和分词器  
2) 注入 LoRA Adapter  
3) 将 Q/A 文本按照模板格式化并分词  
4) 仅对“回答”部分的 token 计算交叉熵损失（使用 mask）  
5) 针对每种风格训练约 50 步  
6) 生成回答并观察风格一致性


## 3. 自定义聊天函数 chat()

chat() 函数实现了：
- 自动构建模型 prompt
- 控制生成长度与温度
- 可选择是否仅返回模型回答部分

用于训练过程中实时观察风格变化。


## 4. 使用 LLM 作为风格判定者（LLM-as-a-Judge）

为了客观评估模型输出是否符合目标风格，本项目使用更大的模型作为“判官”模型：

判官模型：
google/gemma-2-9b-it（通过 OpenRouter 访问）

判官模型接收：
- 风格说明
- 真正的风格样例
- 待评估文本

返回：
{"score": 0 ~ 10}  
最终归一化到 0 ~ 1。

### 评分示例结果
| 文本来源 | 平均得分 | 解释 |
|--------|--------|------|
| 基础英文回答（负对照） | ~0.00 | 不符合 Yoda 风格 |
| 微调后模型生成文本 | ~0.28 | 具有一定 Yoda 风格特征 |
| 真实 Yoda 风格样本（正对照） | ~0.42 | 风格明显，接近标准 Yoda |


## 5. 输出示例

Leprechaun 风格示例：
"Top o' the mornin' to ye! A fine question ye ask! Let me spin ye a tale..."

Yoda 风格示例：
"Hard to see, the answer is. But learn you will, yes."


## 6. 环境与依赖

Python 3.10+
PyTorch
transformers
peft
mitdeeplearning
opik

GPU ≥ 12GB VRAM 推荐。

为避免显存累积，重要设置：
use_cache = False  
torch.cuda.empty_cache()


## 7. 最终评测指标

对保留的 Yoda 测试文本进行对数似然评价：

Yoda test loglikelihood: 2.75

此值可用于不同微调尝试之间的效果比较。


## 8. 致谢

- MIT Deep Learning Labs 提供的教学项目结构
- HuggingFace PEFT (LoRA) 框架
- OpenRouter 提供的更大模型判别接口

### 项目说明
本仓库作为课程实验项目演示 LLM 微调、风格迁移与定量评估方法。

可自由复现、修改、替换为任意新风格。
