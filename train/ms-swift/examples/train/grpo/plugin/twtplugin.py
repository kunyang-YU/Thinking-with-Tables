
import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()



def tat_evaluation(ans, gt):
    gt = gt[0] if type(gt) == list else str(gt)
    if gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ',''):
        return 1
    else:
        return 0


def hitab_evaluation(ans, gt):
    gt = gt[0] if type(gt) == list else str(gt)
    if gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ',''):
        return 1
    else:
        return 0


def is_rounded_equal_interval(x, y, max_k=6):
    for k in range(max_k + 1):
        half = 0.5 * 10**(-k)
        if y - half <= x < y + half:
            return True
    return False

def finqa_evaluation(ans, gt):
    gt = gt[0] if type(gt) == list else str(gt)
    if '%' in gt:
        try:
            ans = float(ans.replace('%',''))
            gt = float(gt.replace('%',''))
        except:
            gt = str(gt)
            ans = str(ans)
    else:
        try:
            gt = float(gt)
            ans = float(ans)
        except:
            gt = str(gt)
            ans = str(ans)
    if  type(gt) != float and (gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or ans.lower().replace(',','').replace(' ','') in gt.lower().replace(',','').replace(' ','') or gt.lower().replace(',','').replace(' ','') in ans.lower().replace(',','').replace(' ','')):
        return 1
    elif type(gt) == float and is_rounded_equal_interval(abs(gt), abs(ans)):
        return 1
    else:
        return 0

def cls_evaluation(ans, gt):
    try:
        gt = int(gt)
        ans = int(ans)
    except:
        return 0
    if gt == ans:
        return 1
    else:
        return 0

def reg_evaluation(ans, gt):
    try:
        gt = float(gt)
        ans = float(ans)
    except:
        return 0
    rel_err = abs(ans - gt) / (abs(gt) + 1e-6)
    reward = 1 - min(rel_err, 1)
    return reward

class CodeFormat(ORM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.warp_func = {
            'HiTab':hitab_evaluation,
            'FinQA': finqa_evaluation,
            'TAT-QA':tat_evaluation,
            'CLS':cls_evaluation,
            'REG': reg_evaluation
        }

    def __call__(self, completions, type, solution, **kwargs) -> List[float]:
        rewards = []
        for content, t, gt in zip(completions, type, solution):
            # print(content)
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, content, re.DOTALL)
            if match is not None:
                answer = match.group(1)
            else:
                answer = '!!!!!!wrong'

            # print(answer, gt)
            
            warp_func = self.warp_func[t]
            answer_reward = warp_func(answer, gt)

            reward = answer_reward
            rewards.append(reward)
        return rewards



import re
import io
import sys
import base64
import pickle
import traceback
import asyncio
from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from swift.plugin import ContextManager, Env, context_managers, envs
from swift.utils import remove_response
import unittest
import multiprocessing as mp
if TYPE_CHECKING:
    from swift.llm.infer.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig,
                                          RolloutOutput)
    from swift.llm.template import RolloutInferRequest
    from swift.llm.infer.infer_engine import GRPOVllmEngine
    from swift.llm.utils import Messages

import io, traceback
from contextlib import redirect_stdout
import signal
import sys
from contextlib import redirect_stdout

class TwTMulltiTurn(MultiTurnScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_env = {}

    def _repl_code_run(self, env, code, timeout=10):
        """使用信号实现超时，不创建新进程"""
        import io, traceback
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError()
        
        buf = io.StringIO()
        result = ""
        
        # 设置超时信号
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            with redirect_stdout(buf):
                lines = code.strip().splitlines()
                if not lines:
                    return ""
                
                *body, last = lines
                try:
                    expr = compile(last.lstrip(), "<string>", "eval")
                    if body:
                        exec("\n".join(body), env)
                    val = eval(expr, env)
                    if val is not None:
                        result = str(val)
                except (SyntaxError, IndentationError):
                    exec(code, env)
        except TimeoutError:
            return "timeout"
        except Exception:
            result = traceback.format_exc()
        finally:
            signal.alarm(0)  # 取消定时器
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
        
        return buf.getvalue() + result


    def _extract_code(self, text):
        code_blocks = re.findall(r"<code>(.*?)</code>", text, re.DOTALL)
        code = "from tools import ResNet50Predict"+"\n".join(code_blocks).replace("```py", "").replace("```", "") # 横线 \n
        return code
    

    def check_finished(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content
        code = self._extract_code(completion)
        code_results = self._repl_code_run(self.code_env, code)
        results_with_tokens = f'\n<code_result>\n{code_results}\n</code_result>\n'
        # print(completion, 'in check')
        if re.search(r'<answer>\s*[\s\S]*?\S[\s\S]*?\s*</answer>', completion) is not None: #and re.search(r'<code>\s*[\s\S]*?\S[\s\S]*?\s*</code>', completion) is None:
            return True
        if completion == '':
            print('finished by zero output')
            # infer_request.messages.append({"role": "user", "content": 'please continue answer the question'})
            return True
        word_leg = ''
        for m in infer_request.messages:
            word_leg += m['content']
        word_leg += results_with_tokens
        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(word_leg, add_special_tokens=False)
        if len(result_tokens) > 10240:
            print('finished due overlength')
            return True
        return super().check_finished(infer_request, response_choice, current_turn)
    
    
    def _extract_body(completion):
        pass

    def _extract_logprobs_from_choice(self, response_choice: 'ChatCompletionResponseChoice') -> List[float]:
        """Extract logprobs list from response choice for rollout importance sampling.

        Args:
            response_choice: The response choice containing logprobs

        Returns:
            List of logprob values, or empty list if not available
        """
        if response_choice.logprobs is None:
            return []
        if 'content' in response_choice.logprobs:
            return [item['logprob'] for item in response_choice.logprobs['content']]
        return []
    
    def step(self, infer_request, response_choice, current_turn):
    
        completion = response_choice.message.content
        # print('------------------------')
        # print(current_turn, completion)
        # print('------------------------')
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)

        code = self._extract_code(completion)
        code_results = self._repl_code_run(self.code_env, code)
        
        results_with_tokens = f'\n<code_result>\n{code_results}\n</code_result>\n'
        # print(results_with_tokens)
        # print(results_with_tokens)
        # infer_request.messages[-1]['content'] += (results_with_tokens)
        infer_request.messages.append({"role": "user", "content": results_with_tokens})
        # # print(infer_request)

        # # print(self.infer_engine)
        # tokenizer = self.infer_engine.default_template.tokenizer
        # result_tokens = tokenizer.encode(results_with_tokens, add_special_tokens=False)
        # token_ids.extend(result_tokens)
        # loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'rollout_infos': {
                'code_results': results_with_tokens,
                'num_turns': current_turn,
            }
        }







orms['re_format'] = CodeFormat
multi_turns['twt_scheduler'] = TwTMulltiTurn