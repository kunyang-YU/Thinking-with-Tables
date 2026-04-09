import re
import io
import sys
import base64
import pickle
import traceback
def sandbox_run(env, code):
    """
    Execute code and collect all results (output, expression values, exception info)
    into the result field.
    Returns:
        {'result': <str>}
    """
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    final_result = ""

    try:
        lines = code.strip().splitlines()
        if not lines:
            return {'result': ''}

        *body, last = lines

        try:
            # Try to compile the last line as an expression
            expr = compile(last, '<string>', 'eval')
            if body:
                exec("\n".join(body), env)

            value = eval(expr, env)
            if value is not None:
                final_result = str(value)

        except SyntaxError:
            # The entire block is statements
            try:
                exec(code, env)
            except Exception:
                final_result = traceback.format_exc()

        except Exception:
            final_result = traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    # Prepend print output to final_result
    output = buffer.getvalue()
    if output:
        final_result = output + final_result

    return final_result

import base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

import argparse
from openai import OpenAI
parser = argparse.ArgumentParser(description="evaluation")
    # Add arguments
parser.add_argument("--exp", type=str, required=True, help="Input file path")
parser.add_argument("--model_name", type=str, required=True, help="Input file path")
parser.add_argument("--port", type=str, required=True, help="Input file path")
args = parser.parse_args()
PORT = args.port
EXP_NAME = args.exp
MODEL_NAME = args.model_name

client = OpenAI(
        api_key="your_api_key",
        base_url=f"http://127.0.0.1:{PORT}/v1",
        timeout=60.0
)
import json
file_path = "your_path/evaluation/multi-modal-evaluation/adoption/adoption-test.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
results_dict = []
from tqdm import tqdm
for id, d in tqdm(enumerate(data)):
    # print('________________')
    try:
        res = {}
        res['id'] = id
        env = {}
        image_paths = d['image_path']
        image_base64 = encode_image(image_paths)
        response = ""
        context = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": d['instructions']},
                ]
        while "<answer>" not in response:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": context,
                    },
                ],
                stream=True,
                stop=['</code>']
            )
            response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    response += content
                if len(response) > 10000:
                    res["ans_model"] = '_wrong!!!!!'
                    res['gt'] = d['answer']
                    results_dict.append(res)
                    print(f"[{id}] Answer:", ans, 'gt', d['answer'])
                    raise ValueError('repeat 2 much')
                # print(response)
            print('response success')#, response)
            # print(response)
            response += '</code>'
            code_blocks = re.findall(r"<code>(.*?)</code>", response, re.DOTALL)
            code = "from tools import ResNet50Predict"+"\n".join(code_blocks).replace("```py", "").replace("```", "")
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
            # print(result)
            # print(response+result_add)
            if len(context) > 15:
                res["ans_model"] = '_wrong!!!!!'
                res['gt'] = d['answer']
                results_dict.append(res)
                print(f"[{id}] Answer:", ans, 'gt', d['answer'])
                raise ValueError('repeat 2 much')
        ans = "".join(re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)).strip()
        res["ans_model"] = ans
        res['gt'] = d['answer']
        results_dict.append(res)
        print(f"[{id}] Answer:", ans, 'gt', d['answer'])
        with open(f"output_{EXP_NAME}_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(d, e)

# with open(f"output_{EXP_NAME}_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
#     json.dump(results_dict, f, ensure_ascii=False, indent=2)