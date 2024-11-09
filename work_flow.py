import warnings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from member_db import create_insurance_db
from typing import Dict, List, Any, Optional
from llama_index.core.tools import BaseTool
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection, LLM
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context
)
from llama_index.utils.workflow import draw_all_possible_flows

vector_db_loc = "./policy_kb"

claim_form_store_location = "./claim_kb"

vector_db_name = "policy_docs"

def initialize_environment():
    warnings.filterwarnings('ignore')
    _ = load_dotenv()
    return OpenAI(model="gpt-4o-mini"), OpenAIEmbedding(model_name="text-embedding-3-large")

def setup_vector_store(embed_model):
    db = chromadb.PersistentClient(path=vector_db_loc)
    chroma_collection = db.get_or_create_collection(vector_db_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model), chroma_collection

def create_query_tools(index):
    doc_query_engine = index.as_query_engine()
    sql_query_engine = create_insurance_db()
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description="Useful for translating a natural language query into a SQL query over a table containing: insurence member details",
        name="sql_tool"
    )

    policy_query_tool = QueryEngineTool.from_defaults(
        query_engine=doc_query_engine,
        description="Useful for answering semantic questions about insurence policies, procedures and filled auto insurence claim forms.",
        name="policy_query_tool"
    )
    return [sql_tool, policy_query_tool]

class InputEvent(Event):
    """Input event."""

class GatherToolsEvent(Event):
    """Gather Tools Event"""
    tool_calls: Any

class ToolCallEvent(Event):
    """Tool Call event"""
    tool_call: ToolSelection

class ToolCallEventResult(Event):
    """Tool call event result."""
    msg: ChatMessage

class RouterOutputAgentWorkflow(Workflow):
    """Custom router output agent workflow."""

    def __init__(self,
        tools: List[BaseTool],
        timeout: Optional[float] = 10.0,
        disable_validation: bool = False,
        verbose: bool = False,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """Constructor."""
        super().__init__(timeout=timeout, disable_validation=disable_validation, verbose=verbose)
        self.tools: List[BaseTool] = tools
        self.tools_dict: Optional[Dict[str, BaseTool]] = {tool.metadata.name: tool for tool in self.tools}
        self.llm: LLM = llm or OpenAI(temperature=0, model="gpt-4o-mini")
        self.chat_history: List[ChatMessage] = chat_history or []
   
    def reset(self) -> None:
        """Resets Chat History"""
        self.chat_history = []
    
    @step()
    async def prepare_chat(self, ev: StartEvent) -> InputEvent:
        message = ev.get("message")
        if message is None:
            raise ValueError("'message' field is required.")    
        chat_history = self.chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        return InputEvent()

    @step()
    async def chat(self, ev: InputEvent) -> GatherToolsEvent | StopEvent:
        """Appends msg to chat history, then gets tool calls."""
        chat_res = await self.llm.achat_with_tools(
            self.tools,
            chat_history=self.chat_history,
            verbose=self._verbose,
            allow_parallel_tool_calls=True
        )
        tool_calls = self.llm.get_tool_calls_from_response(chat_res, error_on_no_tool_call=False)      
        ai_message = chat_res.message
        self.chat_history.append(ai_message)
        if self._verbose:
            print(f"Chat message: {ai_message.content}")
        if not tool_calls:
            return StopEvent(result=ai_message.content)
        return GatherToolsEvent(tool_calls=tool_calls)
    
    @step(pass_context=True)
    async def dispatch_calls(self, ctx: Context, ev: GatherToolsEvent) -> ToolCallEvent:
        """Dispatches calls."""
        tool_calls = ev.tool_calls
        await ctx.set("num_tool_calls", len(tool_calls))
        for tool_call in tool_calls:
            ctx.send_event(ToolCallEvent(tool_call=tool_call))      
        return None
    
    @step()
    async def call_tool(self, ev: ToolCallEvent) -> ToolCallEventResult:
        """Calls tool."""
        tool_call = ev.tool_call
        id_ = tool_call.tool_id
        if self._verbose:
            print(f"Calling function {tool_call.tool_name} with msg {tool_call.tool_kwargs}")
        tool = self.tools_dict[tool_call.tool_name]
        output = await tool.acall(**tool_call.tool_kwargs)
        msg = ChatMessage(
            name=tool_call.tool_name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": tool_call.tool_name
            }
        )
        return ToolCallEventResult(msg=msg)
    
    @step(pass_context=True)
    async def gather(self, ctx: Context, ev: ToolCallEventResult) -> StopEvent | None:
        """Gathers tool calls."""
        tool_events = ctx.collect_events(ev, [ToolCallEventResult] * await ctx.get("num_tool_calls"))
        if not tool_events:
            return None     
        for tool_event in tool_events:
            self.chat_history.append(tool_event.msg)      
        return InputEvent()

def create_workflow():
    draw_all_possible_flows(RouterOutputAgentWorkflow)
    llm, embed_model = initialize_environment()
    index, _ = setup_vector_store(embed_model)
    tools = create_query_tools(index)
    return RouterOutputAgentWorkflow(
        tools=tools,
        verbose=True,
        timeout=120,
        llm=llm
    )

# wf = create_workflow()

# Example usage (commented out)
# async def main():
#     # message_1 = "List all the member details for members with policy type Health"
#     # message_1 = "Whats the cashback amount for dental expenses?"
#     message_1 = "Given the accident that happened on Lombard Street, name a party that is liable for the damages and explain why?"
#     return await wf.run(message=message_1)
    
# if __name__ == "__main__":
#     import asyncio
#     result = asyncio.run(main())
#     print(result)

