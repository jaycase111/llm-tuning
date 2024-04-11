支持多种模型分布式训练数据构建

SFT输入文件规范:
    格式: Json文件 -> List[Dict]
    轮数: 理论无限制、但会随着用户输入Token限制及LLM-Max-Token做截断
    单个数据样例:
            {
        "id": 2,
        "instruction"": "元旦晚会",
        "dialog": [
            {
                "text": "妈妈，今天新年班会很棒，老师放了很酷的歌曲，我们还跳舞唱歌，真的很开心！",
                "speaker": "user"
            },
            {
                "text": "哇，听起来很有趣呢！",
                "speaker": "assistant"
            },
        ]
            }

Reward输入文件规范
    格式: Json文件 -> List[Dict]
    轮数: 因为为Reward数据、所以轮数为1、一问一答数据
    单个数据样例:
                    {
        "id": 2,
        "instruction"": "元旦晚会",
        "dialog": [
            {
                "text": "妈妈，今天新年班会很棒，老师放了很酷的歌曲，我们还跳舞唱歌，真的很开心！",
                "speaker": "user"
            },
            {
                "text": "哇，听起来很有趣呢！",
                "speaker": "assistant"
            },
        ]
        "score": 1.
            }

    

SFT & Reward阶段训练: 当前支持Llama系列模型 | ChatGLM系列模型数据构造器 

RLHF阶段训练支持: 全系列模型PPO学习