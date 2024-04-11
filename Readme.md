DeepSpeed-分布式训练库

支持完整四阶段训练:
    SFT | Reward | RLHF


支持两种架构-多种模型训练
两种架构:
    CAUSAL-LM | Prefix-LM
多种模型:
    Llama 
    Vicua
    Alpaca
    Baize
    ChatGLM
    Dolly
    Falcon
    Guananco
    Openchat

支持多种低资源训练方式:
    AdaLora
    Adapter
    Adapter-Fusion
    Lora
    QLora
    P-Tuning
    Prefix-Tuning
    Prompt-Tuning


支持多种量化方法:
    Int8
    AWQ
    GPTQ
    Squeezellm (TODO)

支持在线推理
    使用VLLM作为推理后端 (TODO)