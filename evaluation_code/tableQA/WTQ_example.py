import argparse
from openai import OpenAI




parser = argparse.ArgumentParser(description="evaluation")

    # Add arguments
parser.add_argument("--exp", type=str, required=True, help="Input file path")
parser.add_argument("--model_name", type=str, required=True, help="Input file path")
parser.add_argument("--port", type=str, required=True, help="Input file path")


args = parser.parse_args()

EXP_NAME = args.exp
MODEL_NAME = args.model_name
PORT = args.port

client = OpenAI(
        api_key="your_api_key",
        base_url=f"http://127.0.0.1:{PORT}/v1",
)

import json

# Set your file path
if EXP_NAME == 'tabmwp':
    file_path = "your_path/evaluation/tabmwp/problems_test1k.json"
else:
    file_path = "your_path/evaluation/wtq/wtq-test.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ans_key_name = 'answer' if EXP_NAME == 'tabmwp' else 'answers'

prompt = '''You are a helpful assistant.

Solve the following table problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed result (wrapped in <code_result></code_result>) can be returned to aid your reasoning and help you arrive at the final answer.

### **Instruction**
1. The image shows a partial sample of the table (df.head()). The image is only to help you understand the table's format.
2. You can write python code to operate the table for solving the answer or obtain the whole table to direct answer the question. The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
3. The way you use the code to operate the table must be robust, because a single word wrong will make the code fail.

### **Output Format (strict adherence required):**

<analy>Your detailed reasoning process should go here.</analy>
<code>(Optional) the code that need to run in the sandbox</code>
<answer>Your final answer to the user's question goes here.</answer>

<image>

### **Question**
{}

### Table file path: {}
'''
path = "your_path/evaluation/tabmwp/tabmwp_csv/{}.csv" if EXP_NAME == 'tabmwp' else 'your_path/evaluation/wtq/wtq_csv/test_{}.csv'
image_path = "your_path/evaluation/tabmwp/tabmwp_img/{}.png" if EXP_NAME == 'tabmwp' else 'your_path/evaluation/wtq/wtq_image/images/test_{}.png'
import re
import io
import sys
import base64
import pickle
def sandbox_run(env, code):
    """
    REPL-style code execution:
    - env: persistent execution environment dict
    - code: multi-line Python code
    Returns:
        - last expression result or output string
    """
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    result = None
    try:
        lines = code.strip().splitlines()
        if not lines:
            return {'output': '', 'result': None}
        *body, last = lines
        try:
            expr = compile(last, '<string>', 'eval')
            if body:
                exec("\n".join(body), env)
            result = eval(expr, env)
        except SyntaxError:
            exec(code, env)
    finally:
        sys.stdout = old_stdout
    return result if result is not None else output.getvalue()


import base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
results_dict = []
if EXP_NAME == 'tabmwp':
    iter_machine = data.items()
else:
    iter_machine = enumerate(data)
from tqdm import tqdm
for id, d in tqdm(iter_machine):
    try:
        res = {}
        res['id'] = id
        env = {}
        image_base64 = encode_image(image_path.format(id))
        response = ""
        context = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt.format(d['question'], path.format(id))},
                ]
        while "<answer>" not in response:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": context,
                    },
                ]
            )
            response = completion.choices[0].message.content
            print('response success')
            code_blocks = re.findall(r"<code>(.*?)</code>", response, re.DOTALL)
            code = "\n".join(code_blocks).replace("```py", "").replace("```", "")
            try:
                result = sandbox_run(env, code)
                result_add = f'<code_result>\n{result}</code_result>\n'
            # print(response+result_add)
                context.append({"type": "text", "text": response+result_add})
            except:
                result = 'Your Code have Error'
                result_add = f'<code_result>\n{result}</code_result>\n'
            # print(response+result_add)
                context.append({"type": "text", "text": response+result_add})
        ans = "".join(re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)).strip()
        res["ans_model"] = ans
        res['gt'] = d[ans_key_name]
        results_dict.append(res)
        # print(f"[{id}] Answer:", ans, 'gt', d[ans_key_name])
    except Exception as e:
        print(d, e)

with open(f"output_{EXP_NAME}_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False, indent=2)
