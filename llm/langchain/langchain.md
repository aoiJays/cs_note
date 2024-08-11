

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

#### 流式输出

```python
import asyncio
# 流式输出
async def stream_words(prompt, llm):
    async for chunk in llm.astream([prompt]):
        words = chunk.split()
        for word in words:
            print(f"{word}", end='', flush=True) 
   

# 运行主函数
if __name__ == "__main__":
    prompt = "讲一个故事，不超过30字"
    llm = Tongyi( model_name="qwen1.5-1.8b-chat", temperature=0.95, top_p=0.7, max_tokens=10, streaming = True)
    asyncio.run(stream_words(prompt, llm))
```



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
# 推荐invoke
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



## Prompts

### PromptTemplate

> [文档：PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)

```python
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate

# 方法1：PromptTemplate构造函数

# 模板定义
template = "一句话介绍一下明日方舟干员{Operators}的身份"
# 构建组件
prompt = PromptTemplate(template=template, input_variables=["Operators"])
# 生成模板
prompt_text = prompt.format(Operators="缪尔赛斯")

# --------------------------------------

# 方法2：from_template
template = "一句话介绍一下明日方舟干员{Operators}的身份"
prompt = PromptTemplate.from_template(template)
prompt_text = prompt.format(Operators="缪尔赛斯")

# --------------------------------------



llm = Tongyi( model_name="qwen1.5-1.8b-chat", temperature=0.95, top_p=0.7, max_tokens=100 )

llm(prompt_text)
```

### FewShotPromptTemplate

> [文档：FewShotPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.few_shot.FewShotPromptTemplate.html#)
>
> - Zero-shot 是指模型在没有见过任何特定任务的训练数据的情况下进行预测
> - One-shot 是指模型在仅见过一个训练样本的情况下进行学习和预测
> - Few-shot 是指模型在见过很少量的训练样本（几个到几十个）的情况下进行学习和预测

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate


# 样例准备
# 相当于会组合样例以及模板 成为真正的样例
examples = [
    {
        "Operators": "令", 
        "description": "令，一位辅助干员"
    },
    {
        "Operators": "斯卡蒂", 
        "description": "斯卡蒂，一位近卫干员"
    }, 
    {
        "Operators": "缪尔赛斯", 
        "description": "缪尔赛斯, 一位先锋干员"
    }
]
example_template = """
明日方舟干员：{Operators}
描述：{description}\n
"""
example_prompt = PromptTemplate.from_template(example_template)


few_shot_prompt = FewShotPromptTemplate(
    examples = examples, 
    example_prompt = example_prompt, 
    
	# 真实提问
    prefix = "给出每个干员的姓名，描述每一位干员的身份",
    suffix = "明日方舟干员：{Operators}\n描述：\n",
    input_variables=["Operators"],
    example_separator = "\n" # 样例直接隔开
)

prompt_text = few_shot_prompt.format(Operators="黍")
print(prompt_text)

'''
# prompt_text:
给出每个干员的姓名，描述每一位干员的身份

明日方舟干员：令
描述：令，一位辅助干员



明日方舟干员：斯卡蒂
描述：斯卡蒂，一位近卫干员



明日方舟干员：缪尔赛斯
描述：缪尔赛斯, 一位先锋干员


明日方舟干员：黍
描述：
'''
print(llm(prompt_text))
'''
# 输出
明日方舟干员：谷粒
描述：谷粒，一位研究者干员

# 至少格式对了
'''
```



## Chains

### LLMChain

> [对象文档：LLMChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#langchain.chains.llm.LLMChain)

```python
from langchain.chains import LLMChain
# 指定模型 指定prompt
chain1 = LLMChain(llm=llm, prompt=few_shot_prompt)
chain1.invoke("黍")
# {'Operators': '黍', 'text': '明日方舟干员：谷粒\n描述：谷粒，一位研究者干员'}
```

需要多个参数时：

```python
# 模板定义
template = "{a} + {b} = ?"
# 构建组件
prompt = PromptTemplate.from_template(template)

chain2 = LLMChain(llm=llm, prompt=prompt)
chain2.invoke({"a": "100", "b": "200"})
# {'a': '100', 'b': '200', 'text': '300'}
```

### SimpleSequentialChain

将上一个chain的输出作为当前chain的输入

```python
template = "{country}的首都是哪个城市？用一个词回答"
prompt = PromptTemplate.from_template(template)
chain1 = LLMChain(llm=llm, prompt=prompt)

template2 = "{city}有哪些景点值得参观？用一句话列举"
prompt2 = PromptTemplate.from_template(template2)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# 组装
from langchain.chains import SimpleSequentialChain
chains = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
# verbose=True，可以看见推理的步骤
chains.run("中国")

'''
> Entering new SimpleSequentialChain chain...
首都：北京。
故宫、天安门广场、长城、颐和园、鸟巢等。
> Finished chain.
'''

'''
# 真实输出
'故宫、天安门广场、长城、颐和园、鸟巢等。'
'''

```

### LLMChain（再解释）

每一个LLMChain，绑定了模型和prompt

成为一个单独的元素



```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# LLMChain1 英文翻译中文
chain1_prompt = PromptTemplate.from_template('''
请将以下文本翻译成流畅的中文：
文本：{txt1}
翻译：
''')
chain1 = LLMChain(llm=llm, prompt=chain1_prompt)

# LLMChain2 中文归纳
chain2_prompt = PromptTemplate.from_template('''
请用简洁的一句话，概括以下文本：
文本：{txt2}
概括：
''')
chain2 = LLMChain(llm=llm, prompt=chain2_prompt)

# LLMChain3 新闻标题
chain3_prompt = PromptTemplate.from_template('''
请为以下文本生成一个简洁且吸引人的新闻标题：
文本：{txt3}
新闻标题：
''')
chain3 = LLMChain(llm=llm, prompt=chain3_prompt)
```



### SimpleSequentialChain（再解释）

- 每个LLMChain的输出，将作为下一个LLMChain的输入

```python
from langchain.chains import SimpleSequentialChain
chains = SimpleSequentialChain(
    chains = [ chain1, chain2, chain3], 
    verbose = True
)

chains.invoke(txt)
```

运行结果：

```python
'''
> Entering new SimpleSequentialChain chain...
近年来，莫尔斯西小姐在哥伦比亚学术界享有极高的声誉。她在不到30岁的时候，就已经成为Rhein Life十强人物之一，担任生态学与环境科学大学副校长一职，并担任特雷蒙多大学客座讲师。过去十年间，莫尔斯西及其研究团队在生态学、遗传学、植物生物化学和植物生理等领域取得了显著突破，对哥伦比亚生命科学和环境科学的发展做出了重要贡献。
莫尔斯西小姐在哥伦比亚学术界享有极高声誉，是知名生态学与环境科学大学副校长及特雷蒙多大学客座讲师，在近十年间在多个领域取得显著突破，对哥伦比亚生命科学和环境科学发展做出重要贡献。
"哥伦比亚学者莫尔斯西小姐：杰出生态学家、客座讲师，近十年里多项研究成果瞩目，引领哥伦比亚生命科学和环境科学发展"

> Finished chain.
'''
```



### SequentialChain

我们需要更加复杂的chain关系

- 第三个LLMChain的输入，由第一个和第二个提供……



我们将对模板以及LLMChain的key非常敏感



```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# LLMChain1 英文翻译中文
chain1_prompt = PromptTemplate.from_template('''
请将以下文本翻译成流畅的中文：
文本：{Source_txt}
翻译：
''')
chain1 = LLMChain(llm=llm, prompt=chain1_prompt, output_key='Translated_txt')


# LLMChain2 语言分类
chain2_prompt = PromptTemplate.from_template('''
这一段文本使用的是哪一种语言？
文本：{Source_txt}
语言：
''')
chain2 = LLMChain(llm=llm, prompt=chain2_prompt, output_key='answer')

# LLMChain3 中文归纳
chain3_prompt = PromptTemplate.from_template('''
请用简洁的一句话，概括以下文本：
文本：{Translated_txt}
概括：
''')
chain3 = LLMChain(llm=llm, prompt=chain3_prompt, output_key='summary')

```



```python
from langchain.chains import SequentialChain
chains = SequentialChain(
    chains = [chain1, chain2, chain3],
    input_variables = ['Source_txt'],
    output_variables = ['Translated_txt','answer', 'summary'], # 如果希望返回中间的输出结果
    verbose=True
)

chains.invoke(txt)
'''
{'Source_txt': 'Miss Muelsyse is a highly regarded genius scientist in the academic circles of Columbia in recent years. At less than thirty years old, she has already become one of the top ten figures at Rhine Life, serving as the Director of Ecology and a guest lecturer at the University of Trimondo. Over the past decade, Miss Muelsyse and her research team have made significant advancements in various fields, including ecology, genetics, plant biochemistry, and plant physiology, greatly contributing to the development of life sciences and environmental sciences in Columbia.',
 'Translated_txt': '近年来，莫尔斯西小姐在哥伦比亚学术界享有极高的声誉。她在不到30岁的时候，就已经成为Rhein Life十强人物之一，担任生态学与环境科学大学副校长一职，并担任特雷蒙多大学客座讲师。过去十年间，莫尔斯西及其研究团队在生态学、遗传学、植物生物化学和植物生理等领域取得了显著突破，对哥伦比亚生命科学和环境科学的发展做出了重要贡献。',
 'answer': '这段文本使用了英语。',
 'summary': '莫尔斯西小姐在哥伦比亚学术界享有极高声誉，是知名生态学与环境科学大学副校长及特雷蒙多大学客座讲师，在近十年间在多个领域取得显著突破，对哥伦比亚生命科学和环境科学发展做出重要贡献。'}
'''
```

## Agents

预训练的LLM能力有限（数学能力、逻辑、实时信息），需要引入第三方工具进行辅助

>   [Toolkits | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/toolkits/)
>
>   [Tools | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/tools/)

```python
!pip install numexpr
from langchain.agents import load_tools, initialize_agent, AgentType

# 加载第三方工具
tools = load_tools(["llm-math"], llm=llm)
# 初始化agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Assume you are 12 years old now, what's your age raised to the 0.43 power?")

'''
To calculate the amount by which my current age is raised to the power of 0.43, we first need to determine my current age and then multiply it by 0.43.
My current age:
Since I am currently 12 years old, my current age is 12.
To find the power 0.43, we divide the current age by 0.43:
Power = Current Age / 0.43
Power = 12 / 0.43
Power ≈ 30.867294153885714
Thought: The calculation has been completed successfully.
Final Answer: My current age raised to the 0.43 power is approximately 30.87.

> Finished chain.
'''
```



## Memory

### ChatMessageHistory

-   记录消息存储记录

```python
from langchain.memory import ChatMessageHistory

# 记录消息存储记录
history = ChatMessageHistory()

history.add_user_message("今天开学") # user
history.add_ai_message("好的，明天见") # ai

history.messages # 查看消息记录

# [HumanMessage(content='今天开学'), AIMessage(content='好的，明天见')]
```



### 长期记忆

我们需要把之前的对话存储下来，可以考虑存储成字典形式

```python
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict

# 注意messages和message是不一样的，前者是列表，后者是单个消息
# 这里使用s


history = ChatMessageHistory()

history.add_user_message("今天开学") # user
history.add_ai_message("好的，明天见") # ai

dicts = messages_to_dict(history.messages)
print(dicts)
'''
[{'type': 'human', 'data': {'content': '今天开学', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': None, 'example': False}}, {'type': 'ai', 'data': {'content': '好的，明天见', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': None, 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}}]

'''
new_history = messages_from_dict(dicts)
print(new_history)
'''
[HumanMessage(content='今天开学'), AIMessage(content='好的，明天见')]
'''


```



### ConversationChains

```python
from langchain.chains import ConversationChain

# 支持记录的对话chain
conversation = ConversationChain(llm=llm, verbose=True)
```



我们可以看一下template

```python
'''
'The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. 
Current conversation: {history}
Human: {input}
AI:
'''
```



每次都会存储之前的历史记录，放入本次的输入，让AI填写新的回复内容

为了防止回复词数过多，稍微修改了一下：

```python
conversation.prompt.template = '''
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history}
Human: {input}
AI:
注意，只允许使用不超过10个字进行简单回答
'''

response = conversation.predict(input='我姐姐明天要过生日，推荐一束生日花束的种类。')
response = conversation.predict(input='她喜欢的颜色是粉色的。')
response = conversation.predict(input='我为什么要买花')


conversation.memory.chat_memory.messages
'''
[HumanMessage(content='我姐姐明天要过生日，推荐一束生日花束的种类。'),
 AIMessage(content='"玫瑰、百合、郁金香、康乃馨、太阳花"。'),
 HumanMessage(content='她喜欢的颜色是粉色的。'),
 AIMessage(content='粉色康乃馨。'),
 HumanMessage(content='我为什么要买花'),
 AIMessage(content='为了庆祝她生日。')]
'''
```



## Indexes

对文档（txt、pdf、md……）进行处理、检索

### TextLoader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader('source.txt')
txt = loader.load() # 加载文档

print(txt)
print(len(txt))

# [Document(metadata={'source': 'source.txt'}, page_content='【代号】缪尔赛思\n【性别】女\n【战斗经验】没有战斗经验\n【出身地】未公开\n【生日】11月3日\n【种族】精灵\n【身高】169cm\n【矿石病感染情况】\n参照医学检测报告，确认为非感染者。')]
# 1
```

### CharacterTextSplitter

-   常将文本切分成多个段
-   切分方式
    -   按字数，简单，但是容易切出没意义的段落
    -   按特殊字符，例如回车符

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator='\n', # 切分符
    chunk_size=10,  # chunk大小
    chunk_overlap=0 # 重叠
)

# 切割成字符串列表
txt_doc = text_splitter.split_text(txt[0].page_content)
print(txt_doc)

# 切割成document对象列表
texts = text_splitter.split_documents(txt)
print(texts)

'''
['【代号】缪尔赛思', '【性别】女', '【战斗经验】没有战斗经验', '【出身地】未公开', '【生日】11月3日', '【种族】精灵', '【身高】169cm', '【矿石病感染情况】', '参照医学检测报告，确认为非感染者。']

[Document(metadata={'source': 'source.txt'}, page_content='【代号】缪尔赛思'), Document(metadata={'source': 'source.txt'}, page_content='【性别】女'), Document(metadata={'source': 'source.txt'}, page_content='【战斗经验】没有战斗经验'), Document(metadata={'source': 'source.txt'}, page_content='【出身地】未公开'), Document(metadata={'source': 'source.txt'}, page_content='【生日】11月3日'), Document(metadata={'source': 'source.txt'}, page_content='【种族】精灵'), Document(metadata={'source': 'source.txt'}, page_content='【身高】169cm'), Document(metadata={'source': 'source.txt'}, page_content='【矿石病感染情况】'), Document(metadata={'source': 'source.txt'}, page_content='参照医学检测报告，确认为非感染者。')]
'''
```



### 向量数据库

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
import os

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
)

db = FAISS.from_documents(documents, embeddings) # 构建数据库
retriever = db.as_retriever( # 转换为检索器 返回最相关的k个文档
    search_kwargs = {
        'k': 2
	}
)

result = retriever.get_relevant_documents("学院共有几次晋级ICPC国际大学生程序设计竞赛全球总决赛，分别是哪几年？")
result

```

