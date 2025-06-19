## LLM_gptnoe.py
    If you need to use GPT-Neo, please use this file, as its maximum input length is only 2048 tokens. Therefore, it has high requirements for the input length.

## LLM_others.py
    For the other models mentioned in the paper, since the minimum input length is 8096, we did not encounter any input length limitations in our experiments. Therefore, this file can be used.

## LLM_API.py
    If your resources are limited and you cannot run large models locally, you can use the API format. Note that you may need to replace the access address and the **API key** with your own.