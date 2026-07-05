from langchain_ollama import ChatOllama

from config.settings import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)


def use_value(value, default_value):
    # Use the argument value first, then the default/settings value.
    if value is not None:
        return value

    return default_value


def load_llm(
    model_name=None,
    temperature=None,
    num_ctx=None,
    num_predict=None,
    top_p=None,
    repeat_penalty=None,
):
    # Create a LangChain Ollama chat model.
    return ChatOllama(
        model=use_value(model_name, LLM_MODEL_NAME),
        temperature=use_value(temperature, LLM_TEMPERATURE),
        num_ctx=use_value(num_ctx, LLM_NUM_CTX),
        num_predict=use_value(num_predict, LLM_NUM_PREDICT),
        top_p=use_value(top_p, LLM_TOP_P),
        repeat_penalty=use_value(repeat_penalty, LLM_REPEAT_PENALTY),
    )
