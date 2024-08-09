

# Langchain · 二阶段

[TOC]

## Chains

### LLMChain

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



### SimpleSequentialChain

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



> 有用到的时候再补充
