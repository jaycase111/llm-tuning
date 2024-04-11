部分参数微调训练方法
低资源微调方法
    Lora
    QLora
    AdaLora
    Adapter             # 实现基础版本
    # https://blog.csdn.net/2301_77818837/article/details/135355919
    AdpaterFusion       # Peft 中已经废弃 
    https://github.com/adapter-hub/adapters/blob/master/notebooks/03_Adapter_Fusion.ipynb
    https://github.com/adapter-hub/adapters 
    后续研究如何接入该代码
    AdapterDrop         # 原理过于简单、不实现
    MAM-Adapter         # 原理过于简单、不实现
    prefix-tuning       # https://zhuanlan.zhihu.com/p/629327372
    prompt-tuning
    p-tuing V1
    p-Tuning V2 


本库主要调用的高效微调库为:
    PEFT 以及 Adapters