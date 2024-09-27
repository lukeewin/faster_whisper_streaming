# 1. 项目说明
这是一个基于faster whisper实现的仿流式语音识别的项目，其实是一句话识别
当我们讲完一句话的时候，会自动调用 asr 进行语音识别
本项目可以轻松对接大模型，从而实现离线环境的人机对话
# 2. 项目部署
项目使用 python 开发，需要安装 python 环境，以及需要执行下面命令安装项目用到的依赖
```shell
pip install -r requirements.txt
```
运行语音识别使用下面命令运行
```shell
python app.py
```
如果想要运行对接大模型，必须要先安装 ollama，如何安装 ollama 可以到 https://github.com/ollama/ollama 中查看
```shell
python asr_gpt.py
```
运行上面命令中默认使用的大模型是千问1.8b 大模型，如果需要替换为其它模型，可以修改源代码，找到下面的源代码中 model 的值，修改为你需要用的大模型，例如下面中的 "model": "qwen:1.8b"
```python
def gpt(question):
    url = "http://localhost:11434/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    dialogue_history.append({
        "role": "user",
        "content": question
    })
    data = {
        "model": "qwen:1.8b",
        "messages": dialogue_history,
        "stream": False
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        ai_answer = response_json["message"]["content"]
        dialogue_history.append({
            "role": "assistant",
            "content": ai_answer
        })
        return ai_answer
```
运行效果可以到我的 B 站中观看 https://www.bilibili.com/video/BV1fosZe7EBt
# 3. 其它
博客技术分享博客平台：https://blog.lukeewin.top 或者 https://lukeewin.top
我的 B 站平台：https://space.bilibili.com/674558378
有任何问题需要联系我的，可以到我的博客中的留言板中留言