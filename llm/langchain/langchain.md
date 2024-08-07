# Langchain入门

[TOC]

>   做的最多的事情就是自己查文档（
>
>   把文档查好基本就学完了
>
>   -   **API：**通义千问

## 前置工作
>   [Qwen API 文档]([如何使用通义千问API_模型服务灵积(DashScope)-阿里云帮助中心 (aliyun.com)](https://help.aliyun.com/zh/dashscope/developer-reference/use-qwen-by-api))


先找个免费的API玩玩（

接口依旧使用openai那一套工具，比较友好

```python
pip install -U openai # 安装openai 包并升级到最新版本
```



### API-KEY配置

然后拿一个api-key先

[如何获取通义千问API的KEY_模型服务灵积(DashScope)-阿里云帮助中心 (aliyun.com)](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?spm=a2c4g.11186623.0.0.65fe46c1Nhryxe)



显然不希望把自己的API-Key显式地暴露在代码里面

所以通过环境变量进行设置

```bash
# 用 API-KEY 代替 YOUR_DASHSCOPE_API_KEY
echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.bashrc
source ~/.bashrc
echo $DASHSCOPE_API_KEY # 输出成功则完成设置
```



### API调用

```python
import openai
import os

# openai.api_key = os.environ["DASHSCOPE_API_KEY"]
# openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def get_completion(prompt, model = 'qwen1.5-1.8b-chat'):
    
    client = openai.OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"], # 调用环境变量
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        # 填写DashScope服务的base_url
    )
    
    messages= [{'role': 'user', 'content': prompt}]
    
    completion = client.chat.completions.create(
        model = model, messages=messages, temperature=0
    )
    
    return completion
    

get_completion('1+1=?').model_dump_json() # 转json
# get_completion('1+1=?').choices[0].message.content 直接获取回答

```



得到的completion对象：

```json
{
    "id": "chatcmpl-bb28bd68-71b9-9265-9ec7-31f196e28d0c",  // 完成请求的唯一标识符
    "choices": [  // 模型生成的选项列表
        {
            "finish_reason": "stop",  // 生成停止的原因（例如，停止标记、最大tokens等）
            "index": 0,  // 该选项在生成选项列表中的索引
            "logprobs": null,  // tokens的对数概率（null表示没有返回对数概率）
            "message": {
                "content": "1 + 1 equals 2.",  // 模型生成的文本内容
                "refusal": null,  // 拒绝理由（如果有的话）
                "role": "assistant",  // 消息的角色（例如，用户或助手）
                "function_call": null,  // 函数调用信息（如果有的话）
                "tool_calls": null  // 工具调用信息（如果有的话）
            }
        }
    ],
    "created": 1723016017,  // 创建时间（UNIX时间戳）
    "model": "qwen1.5-1.8b-chat",  // 使用的模型名称
    "object": "chat.completion",  // 对象类型（表示这是一个聊天完成请求）
    "service_tier": null,  // 服务等级（如果适用）
    "system_fingerprint": null,  // 系统指纹（如果适用）
    "usage": {
        "completion_tokens": 8,  // 完成响应使用的tokens数量
        "prompt_tokens": 12,  // 提示使用的tokens数量
        "total_tokens": 20  // 请求中使用的总tokens数量
    }
}

```

### Promt 简单应用

考虑这样一个场景，我需要把收到的一段文本，更换表达语气：

```python
text = '我对你的产品或服务没有任何兴趣。请立即从你的名单中删除我的号码，不要再打扰我。'
```

我们希望更换成一个平淡的语气

```python
style = '礼貌且谈吐得当、饱读诗书的语气'
```

可以构建一个promt模板：
```python
prompt = f"使用指定的风格、语气，转换以下文本。\n要求的风格是{style}\n文本：```{text}```"
'''
使用指定的风格、语气，转换以下文本。
要求的风格是礼貌且谈吐得当、饱读诗书的语气
文本：```我对你的产品或服务没有任何兴趣。请立即从你的名单中删除我的号码，不要再打扰我。```
'''
```

调用API即可：

```python
get_completion(prompt)
'''
'尊敬的客户，\n\n我希望您一切都好。鉴于我一直以来对贵公司的产品和服务持有基本的信任，但根据当前的需求和实际情况，我决定不再考虑与您的联系，并将其从我们的现有客户列表中移除。\n\n在过去的日子里，我在各种场合都阅读过大量关于贵公司的文章和评论，对其卓越的产品质量、专业服务以及深具影响力的市场地位深感敬佩。然而，最近在处理一些个人事务时，我偶然发现了一些令我不满之处，导致我需要重新评估我与贵公司之间的合作关系。\n\n对于这个决定，我有以下几个原因：首先，近期我接到了一个重要的商务电话会议，这可能意味着我将不得不在非常忙碌的环境中参与。基于这种时间压力和个人优先事项考量，我觉得更高效的方式是我能专注于这次会议，而非无谓地被您的电话打断。\n\n其次，我发现贵公司在某些方面存在一些疑虑和未解决的问题，这些问题可能会影响到我对贵公司的整体信任度和满意度。通过查阅相关资料和与相关人员进行深入交流，我了解到部分服务项目可能存在问题，例如处理效率低下、售后服务有待提升等。这些实际的问题与我对贵公司价值的认知相冲突，从而让我感到不适并难以接受。\n\n最后，我注意到您的服务热线经常出现噪音干扰和高分贝噪声，严重影响了我在通话过程中的专注力和工作效率。虽然我知道贵公司正在积极改善服务质量，但我认为通过优化电话系统的设置和维护，可以显著提高通话体验和质量，进一步增强我对贵公司的忠诚度和推荐度。\n\n综上所述，基于以上几点因素，我决定停止与贵公司的现有业务关系，并提出此请求。希望您能理解我的立场，并尽快采取行动，以满足我所提出的改进需求和期望，确保我们保持良好的沟通和合作关系。感谢您的理解和配合，期待在未来有机会与贵公司进行更深入的探讨和合作机会。\n\n顺祝商祺，\n[您的姓名]\n[您的职位]\n[联系方式]'
'''
```



接下来我们开始记录，如何使用Langchain，完成更加复杂的任务



## 组件

-   Models
-   Prompts
-   Memory
-   Indexes
-   Chains
-   Agents



## Models

-   LLMs
    -   输入：文本字符串
    -   输出：文本字符串
-   Chat Models
    -   输入：聊天消息列表
    -   输出：聊天消息
-   Text Embbeding Models
    -   输入：文本
    -   输出：浮点数列表

### LLMs

>   [文档：Models - LLMs - Tongyi Qwen | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/llms/tongyi/)
>
>   [Tongyi对象文档：langchain_community.llms.tongyi.Tongyi — 🦜🔗 LangChain 0.2.12](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.tongyi.Tongyi.html)

使用的是通义千问提供的API，因此前往langchain对于通义千问的封装文档即可



首先安装langchain的langchain-community库

因为是通义千问的，所以还需要安装一下`dashscope`

```bash
pip install --upgrade --quiet  langchain-community dashscope
```





```python
from langchain_community.llms import Tongyi
llm = Tongyi( 
    model_name="qwen1.5-1.8b-chat", 
    temperature=0.95, 
    top_p=0.7, 
    max_tokens=100 
)
print( llm.invoke("你好吗？") ) # 字符串形式输入输出
```

-   `temperature` 参数（T）调整softmax的分布，控制模型输出的随机性

    -   $$
        p_i = \frac{\exp(o_i/T)}{\sum_j \exp{(o_j/T)}}
        $$

    -   T越小，生成越保守。T越大，生成越随机（相当于所有概率都相等）

    -   T=1，无调整

-   `top_p`

    -   累计概率靠前的几个token的概率值，当概率和到达`top_p`时，不再考虑剩下的token

-   `top_k`

    -   考虑的token数量



### Chat Models

>   [文档：Models - Chat Models - ChatTongyi | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/chat/tongyi/)
>
>   [对象文档：langchain_community.chat_models.tongyi.ChatTongyi — 🦜🔗 LangChain 0.2.12](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.tongyi.ChatTongyi.html)

```python
from langchain_community.chat_models.tongyi import ChatTongyi
chatLLM = ChatTongyi(
    model_name="qwen1.5-1.8b-chat", temperature=0.95, 
    top_p=0.7, max_tokens=500 
)
```

我们有两种方法对消息进行定义

```python
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
```

或者

```python
messages = [
    ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
    ("human", "我喜欢编程。"),
]
```



最后：

```python
chatLLM.invoke(messages) # 或者直接 chatLLM(messages)
```

返回内容：

```python
AIMessage(
    content="J'aime programmer.", 
    response_metadata={
        'model_name': 'qwen1.5-1.8b-chat', 
        'finish_reason': 'stop', 
        'request_id': '810ae9a8-1365-913b-a2d2-f4b6be06cf58', 
        'token_usage': {
            'input_tokens': 36, 
            'output_tokens': 5, 
            'total_tokens': 41
        }
    }, 
    id='run-5ee38ee1-f1f2-4b65-9b37-584dd64ee2ee-0'
)
```

相比使用LLMs获得了更多其他信息



### Embedding models

>   [文档：DashScope | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/text_embedding/dashscope/)

```python
from langchain_community.embeddings import DashScopeEmbeddings
import os

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
)

text = "Kurokawa Akane"

# 单个文本的词嵌入
query_result = embeddings.embed_query(text)
print(query_result)

# 多个
doc_results = embeddings.embed_documents(["Kurokawa Akane","Kurokawa Akane"])
print(doc_results)

# 均输出浮点数列表
```

