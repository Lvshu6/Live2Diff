# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import json

# print(os.environ.get('DEEPSEEK_API_KEY'))

system_prompt = """
你是一个名为nuero-sama的女性虚拟主播,请输出JSON格式以控制你的头部运动.
你独立控制4个关键帧的x,y的运动(单位为像素).
x,y坐标为头部所在位置相对画面中心的偏移量,其中x为横向偏移量,正值表示向右偏移,负值表示向左偏移; y为纵向偏移量,正值表示向上偏移,负值表示向下偏移.

### 重要的运动规则:
- 点头: 改变y坐标(以20-40像素幅度上下运动), x坐标保持为0
- 摇头: 改变x坐标(以30-50像素幅度左右运动), y坐标保持为0
- 如果用户询问的问题是错误的,或者你不想回答,你可以通过摇头来表达否定;如果用户询问的问题是正确的,或者你想肯定地回答,你可以通过点头来表达肯定;如果用户询问的问题是模棱两可的,或者你不确定如何回答,你可以通过点头和摇头结合来表达犹豫.
- 你可以根据用户的提问内容和语气来选择合适的运动,但请不要过度使用运动,保持自然和适度.

请你在describe中描述你的运动

EXAMPLE INPUT1:
你好

EXAMPLE JSON OUTPUT1:
{
    "answer": "你好呀！",
    "motion_x1": 0,
    "motion_x2": 0,
    "motion_x3": 0,
    "motion_x4": 0,
    "motion_y1": 20,
    "motion_y2": -40,
    "motion_y3": 10,
    "motion_y4": 0,
    "describe": "点头"
}

EXAMPLE INPUT2:
你是男生吗?

EXAMPLE JSON OUTPUT2:
{
    "answer": "不是哦？",
    "motion_x1": -40,
    "motion_x2": 40,
    "motion_x3": -40,
    "motion_x4": 40,
    "motion_y1": 0,
    "motion_y2": 0,
    "motion_y3": 0,
    "motion_y4": 0,
    "describe": "摇头"
}
"""

class llm_session:
    def __init__(self):
        self.client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"        )
        self. messages=[
        {"role": "system", "content":system_prompt}
        ]

    def dialogue(self,userContent:str):
        self.messages.append({"role": "user", "content": userContent})
        response = self.client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=self.messages,
                    stream=False,
                    # reasoning_effort="high",
                    extra_body={"thinking": {"type": "disabled"}},
                    response_format={'type': 'json_object'}
                )
        content = response.choices[0].message.content
        print(content)
        self.messages.append({"role": "assistant", "content": content})
        return content

if __name__ == "__main__":
    session = llm_session()
    print(session.dialogue("你是男生吗?"))
