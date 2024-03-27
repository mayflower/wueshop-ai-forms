import os
import re
import textwrap
from typing import Any, AsyncIterator, Dict, List, Tuple

import pandas as pd
import torch
from bqplot import ColorScale, LinearScale
from dotenv import load_dotenv
from ipydatagrid import BarRenderer, DataGrid
from langchain._api import suppress_langchain_deprecation_warning
from langchain.evaluation import load_evaluator
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langgraph.channels.base import ChannelsManager
from langgraph.pregel import Pregel, _prepare_next_tasks
from transformers import AutoModelForMaskedLM, AutoTokenizer

load_dotenv()


def llm(temperature: float = 0.7, model: str = None, streaming: bool = True, **kwargs):
    """
    wrapper around the OpenAI and Azure OpenAI LLMs. Takes per default the model where the KEY is set in the environment variables.
    First takes OpenAI if the key is set.

    Args:
        temperature (float, optional): temperature for sampling. Defaults to 0.7.
        model (str, optional): model name. Defaults to None. Specify OpenAI model string or Azure deployment name. Otherwise tries to get the model from the environment variables.
    """
    if os.environ.get("OPENAI_API_KEY"):
        model_name = model if model else os.environ.get("OPENAI_MODEL")
        return ChatOpenAI(
            model=model_name, temperature=temperature, streaming=streaming, **kwargs
        )
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        deployment = model if model else os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            temperature=temperature,
            streaming=streaming,
            **kwargs,
        )
    else:
        raise ValueError("No provider secret found in environment variables.")


def embeddings(model=None, **kwargs):
    if os.environ.get("OPENAI_API_KEY"):
        model_name = model if model else os.environ.get("OPENAI_EMBEDDING")
        return OpenAIEmbeddings(model=model_name, **kwargs)
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        deployment = model if model else os.environ.get("AZURE_OPENAI_EMBEDDING_NAME")
        return AzureOpenAIEmbeddings(azure_deployment=deployment, **kwargs)
    else:
        raise ValueError(" No provider secret found in environment variables.")

