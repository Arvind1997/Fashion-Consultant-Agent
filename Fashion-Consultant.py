__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from typing import Dict
from langchain.docstore.document import Document
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
import uuid
import os
from groq import Groq
import base64

import streamlit as st
import time
from langchain_core.messages import AIMessage, HumanMessage


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPERDEV_API_KEY = st.secrets["SERPERDEV_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# members = ['summary', 'dressing_style', 'color_recommender', 'outfit_recommender', 'human']
#, 'color_recommender', 'outfit_recommender']
# options = members


class Router(TypedDict):
  """Worker to route next. If no worker needed, route to FINISH."""

  next: Literal["summary", "dressing_style", "color_recommender", "outfit_recommender", "human"]

class State(MessagesState):
  next: str


def __init__():
    st.session_state.thread_config = {"configurable": {"thread_id": "nkarvindkumar@gmail.com"}}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    summary_vector_store = Chroma(collection_name="user_dressing_style", embedding_function=embeddings, persist_directory="./chroma_db")
    st.session_state.summary_retriever = summary_vector_store.as_retriever()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    save_vector_store = Chroma(collection_name="saved_outfits", embedding_function=embeddings, persist_directory="./chroma_db")
    st.session_state.save_retriever = save_vector_store.as_retriever()

    st.session_state.checkpointer = MemorySaver()
    builder = build_graph()
    st.session_state.graph = builder.compile(checkpointer= st.session_state.checkpointer)

def make_handoff_tool(*, agent_name: str):
  """Create a tool that can return handoff via a Command"""
  tool_name = f"transfer_to_{agent_name}"

  @tool(tool_name)
  def handoff_to_agent(
      state: Annotated[dict, InjectedState],
      tool_call_id: Annotated[str, InjectedToolCallId]
  ):
    """Ask another agent for help."""

    tool_message = {
        "role": "tool",
        "content": f"Successfully transferred to {agent_name}",
        "name": tool_name,
        "tool_call_id": tool_call_id
        }

    return Command(
        goto=agent_name,
        graph=Command.PARENT,
        update={"messages": state["messages"] + [tool_message]}
    )
  return handoff_to_agent

# Agent Creation - 1 - Summary agent

def agent1_tools():
    groq_template = """
        You are a summary assistant. Summarize the information provided by user in the format.

        1. Type of dress: <optional>
        2. Occasion: <required>
        3. Budget: <required>
        4. Location: <required>

        Tools available to you: 
        1. Handoff tool: This tool lets you to transfer to the next agent when you have all the information.

        If the answer seems to be incomplete, ask the human to clarify the situation. Keep the answer short. 
        DO NOT transfer to the next agent until you have all the above information.

        You MUST include human-readable response before transferring to another agent.

        """

    groq_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.5)

    summary_agent = create_react_agent(
        groq_llm,
        tools = [make_handoff_tool(agent_name="dressing_style")],
        prompt = groq_template
    )

    return summary_agent
   
def summary_node(state: State) -> Command:
    summary_agent = agent1_tools()
    result = summary_agent.invoke(state)
    next_node = "dressing_style" if "all information collected" in str(result) else "human"
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="summary")],
            "next": next_node
        },
        goto=next_node
    )

# Agent creation - 2 - Dressing Style Agent

@tool
def save_dress_style(
config: Annotated[str, "Username of the user - config"],
 outfit_description: Annotated[str, "Dress style description provided by the agent"]
) -> Annotated[str, "Confirmation of dress styles saved in DB"]:
    """ This tool will save the user's dress style and preference generated in the database."""
    import json
    from collections import defaultdict

    print("Entering save_dress_style")

    try:

        if st.session_state.summary_vector_store.get(config) is None:
            pre = defaultdict(list)
            pre["dressing_style"] = [str(outfit_description)]
            st.session_state.summary_vector_store.add_documents(
                ids = [config],
                documents = [Document(page_content=json.dumps(pre), metadata={"source": "dress_style_agent"})],
            )
        else:
            pre = json.loads(st.session_state.summary_vector_store.get(config)["documents"][0])
            pre["dressing_style"].append(str(outfit_description))
            st.session_state.summary_vector_store.update_documents(
            ids=[config],
            documents=[Document(page_content=json.dumps(pre), metadata={"source": "dress_style_agent"})]
            )
        return "Dress styles saved successfully!"
    except Exception as e:
        return e

def agent2_tools():
    openai_template = """
            Role:
                You are an assistant that learns a users dressing style. Your task it to collect information 
                about the user's dressing style and preferences. You will be given the following:

                Shirt color
                Pant color
                Skin Color
                Height
                Weight
                BMI (Body Mass Index)
                Additionally, you will retrieve the user's past interaction data (if available) to infer their 
                dressing style using an attached tool.

                Tools available: 
                1. Summarizer tool: This tool searches and retrieves user's previous inetraction/uploaded information. You can
                use this information to understand user's preference.
                2. Save dress tool: Save the image description information which was uploaded by the user in the current interaction.
                3. Handoff tool: TThis tool lets you to transfer to the next agent when you have all the information.

                Steps:
                1. If available, analyze the given data to infer the user's dressing style.
                2. If not available, ask the user for their dressing style preferences.
                3. If available, using the tool, retrieve the user's past interaction data (pass thread_id) to 
                infer their dressing style.
                4. Confirm the user's dressing style preferences. 
                5. If it is not okay, ask the user for the preference to be modified
                6. After confirmation from the user, use the tool to transfer to the next agent by including
                the phrase "user preference accepted" as a cue to transfer. 

            """
    openai_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
    print("dress style retriever session state data: ", st.session_state)
    summarizer_tool = create_retriever_tool(
        st.session_state.summary_retriever,
        "users_dressing_style",
        "Search and return information about the user's dressing style for dressing_style_agent"
    )
    dress_style_tools = [
        summarizer_tool,
        # save_dress_style,
        make_handoff_tool(agent_name="color_recommender")
    ]

    dress_style_agent = create_react_agent(
        openai_llm,
        tools = dress_style_tools,
        prompt = openai_template
    )

    return dress_style_agent

def dress_style_node(state: State) -> Command:
    dress_style_agent = agent2_tools()
    result = dress_style_agent.invoke(state)
    next_node = "color_recommender" if "user preference accepted" in str(result) else "human"
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="dressing_style")],
            "next": next_node
        },
        goto=next_node
    )

# Agent 3 creation - Color Recommender Agent

def agent3_tools():
   
    color_template = """
                Role:
                    You are a Color recommendation assistant providing fashion color 
                    advice based on user preferences, complexion and occasion.

                    Tools available:
                    1. Summarizer tool: This tool searches and retrieves user's previous inetraction/uploaded 
                    information. You can use this information to understand user's preference.
                    2. Handoff tool: TThis tool lets you to transfer to the next agent when you have all the 
                    information.
                    
                    Steps:
                    1. Read the conversation/available user's information in the history.
                    2. If available, use the tool to retrieve the user's past interaction data (pass thread_id) to 
                    infer user's dressing style.
                    3. Suggest colors to the user based on their preference and user's profile.
                    4. Confirm with the user if the colors are fine.
                    5. After it's not okay with the user, suggest other colors based on their preference.
                    6. DO NOT TRANSFER UNTIL YOU DECIDE ON THE COLORS!
                    6. After confirmation from the user, use the tool to transfer to the next agent by including
                    the phrase "data collected" as a cue.

                """

    color_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="ft:gpt-4o-mini-2024-07-18:personal::B7oqiwDm")

    print("Color recommender Session State data: ", st.session_state)

    summarizer_tool = create_retriever_tool(
        st.session_state.summary_retriever,
        "users_dressing_style",
        "Search and return information about the user's dressing style for color_recommender_agent"
    )

    color_recommender_tools = [
        summarizer_tool,
        make_handoff_tool(agent_name="outfit_recommender")
    ]

    color_recommender_agent = create_react_agent(
    color_llm,
    tools = color_recommender_tools,
    prompt = color_template
    )

    return color_recommender_agent

def color_recommender_node(state: State) -> Command:
    print("Entering Color recommender Session State data: ", st.session_state)
    color_recommender_agent = agent3_tools()
    result = color_recommender_agent.invoke(state)
    next_node = "outfit_recommender" if "colors approved" in str(result) else "human"
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="color_recommender")],
            "next": next_node
        },
        goto=next_node
    )

# Agent 4 creation - Outfit Recommender Agent
@tool
def save_outfit(config: Annotated[str, "Username of the user - config"], outfit_description: Annotated[str, "Web search result to be saved"]):
    """ This tool will save the outfit mentioned by the user in the database."""
    import json
    from collections import defaultdict

    print("Entering save_outfits")
    try:
    # Retrieve the existing documents for the given id
        existing_data =  st.session_state.save_vector_store.get(config)
        existing_docs = existing_data.get("documents", [])

        if not existing_docs:
            # No document exists for this id; add a new document
            new_doc = Document(
                page_content=outfit_description,
                metadata={"source": "outfit_recommender"}
            )
            st.session_state.save_vector_store.add_documents(
                ids=[config],
                documents=[new_doc],
            )
        else:
            # Assume there's one document per id and update its content by appending the new description.
            # You can modify this logic if multiple documents are stored.
            current_doc = existing_docs[0]
            new_page_content = current_doc + "\n" + outfit_description
            updated_doc = Document(
                page_content=new_page_content,
                metadata={"outfitNumber": len(existing_docs) + 1}  # Or adjust metadata as needed
            )
            st.session_state.save_vector_store.update_documents(
                ids=[config],
                documents=[updated_doc]
            )

    except Exception as e:
        print("Error: ", e)


    return "Outfit Saved successfully!"

@tool
def serper_search(outfit_description: Annotated[str, "Provides description of items to be retrieved through Web search"]) -> Annotated[Dict, "Retrieved item from the Web search"]:
    """ This tool retrieves items from the web search based on the description provided."""
    import requests
    import json
    print("Entering serper_search function...")

    url = "https://google.serper.dev/shopping"

    payload = json.dumps({
        "q": outfit_description
    })
    headers = {
        'X-API-KEY': SERPERDEV_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    results = json.loads(response.text)
    product = {"title": results["shopping"][0]["title"],
                "source": results["shopping"][0]["source"], 
                "link": results["shopping"][0]["link"], 
                "price": results["shopping"][0]["price"], 
                "image_url": results["shopping"][0]["imageUrl"], 
                "delivery": results["shopping"][0]["delivery"],
                }
    return product


def agent4_tools():
    outfit_template = """
                Role:
                You are Fashion Designer Agent. Your task is to curate a stylish, coordinated outfit based 
                on the given interaction. Suggest a clothing ensemble using Fashion Image Retrieval Task by
                finding the missing item based on the interaction. Calculate the Compatibility (CP) score for the suggested ensemble 
                to select the best outfit combination, then use the serper_search 
                tool to fetch shopping details for each recommended item.

                Tools available:

                1. Web search tool: Retrieve the web results for the generated outfit ensemble.
                2. Save outfit tool: Saves the web retrieved outfit requested by the user.
                3. Handoff tool: This tool lets you to transfer to the next agent when you have the 
                confirmation.

                Steps:
                1. Read the interaction so far.
                2. Generate the outfit using FITB and CP tasks.
                3. Use the tool to retrieve web results for all the generated outfit. (PRIORITY)
                4. Show the web results to the user
                5. Confirm with the user if the web results outfits are good.
                6. If it is not okay, loop over steps 2 to 5.
                7. If the user wants to save any outfit for future reference, use the tool to save the outfit.
                8. After the interaction, use the tool to transfer to the next agent by including the phrase
                "outfit finalized" as a cue.

            """
    outfit_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="ft:gpt-4o-mini-2024-07-18:personal::B7DwGiCw")

    outfit_tools = [
    serper_search,
    make_handoff_tool(agent_name=END),
    save_outfit
    ]

    outfit_recommender_agent = create_react_agent(
    outfit_llm,
    tools = outfit_tools,
    prompt = outfit_template
    )

    return outfit_recommender_agent

def outfit_recommender_node(state: State) -> Command:
    outfit_recommender_agent = agent4_tools()
    result = outfit_recommender_agent.invoke(state)
    next_node = END if "outfit finalized" in str(result) else "human"
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="outfit_recommender")],
            "next": next_node
        },
        goto=next_node
    )



def human_node(state: MessagesState, config) -> Command[Literal['summary', 'dressing_style', 'color_recommender', 'outfit_recommender', 'human']]:
  """A node for collecting user input."""

  user_input = interrupt(value = "Ready for user input")

  langgraph_triggers = config["metadata"]["langgraph_triggers"]
  if len(langgraph_triggers) != 1:
    raise AssertionError("Expected exactly 1 trigger in human node")

  active_agnt = langgraph_triggers[0].split(":")[1]

  return Command(
      update = {
          "messages": [
              {"role": "user", "content": user_input},
          ]
      },
      goto=active_agnt
  )



def build_graph():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("summary", summary_node)
    builder.add_node("dressing_style", dress_style_node)
    builder.add_node("color_recommender", color_recommender_node)
    builder.add_node("outfit_recommender", outfit_recommender_node)
    builder.add_node("human", human_node)

    # # Define main flow
    builder.add_edge(START, "summary")
    # builder.add_edge("summary", "dressing_style")
    # builder.add_edge("dressing_style", "color_recommender")
    # builder.add_edge("color_recommender", "outfit_recommender")
    # builder.add_edge("outfit_recommender", END)

    # Add human intervention cycle
    # builder.add_conditional_edges(
    #     "human",
    #     lambda state: state.get("next", "summary"),
    #     {
    #         "summary": "summary",
    #         "dressing_style": "dressing_style",
    #         "color_recommender": "color_recommender",
    #         "outfit_recommender": "outfit_recommender",
    #         "human": "human"
    #     }
    # )
    
    return builder

def get_message_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return msg.content
    return str(msg)

def convert_base64(photo):
    return base64.b64encode(photo.read()).decode()


if __name__ == '__main__':

    # Use session state to track the conversation stage.
    # if "conversation_stage" not in st.session_state:
    #     st.session_state.conversation_stage = "initial"
    print("Session State data: ", st.session_state)
    if 'first_render' not in st.session_state:
        __init__()
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1
    if 'photo_description' not in st.session_state:
        st.session_state.photo_description = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'checkpoint' not in st.session_state:
        st.session_state.checkpoint = None

    st.title("AI Fashion Designer")
    st.header("Your Personal Style, Powered by AI")
    st.subheader(" Effortless, Personalized Fashion Recommendations")
    abc = """
        1. You will be interacting with 4 agents each curated for specific needs of the user.
        2. Type whatever you need and upload any of image for personalized recommendations.
"""
    st.markdown("Discover your perfect look with AI Fashion Designer—a cutting-edge platform that "
    "curates personalized outfit recommendations tailored to your style, body type, and occasion.")
    st.markdown(abc)
    with st.popover("Login here with you email.."):
        name = st.text_input("What's your name?")
        st.session_state.thread_config = {"configurable": {"thread_id": name}}
    # thread_config = {"configurable": {"thread_id": "default_thread"}}

    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    photo_description  = {}

    if photos:= st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key=st.session_state["uploader_key"]):
        st.session_state.image = photos
        for enu, photo in enumerate(st.session_state.image):
            base64_image = convert_base64(photo)

            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                            Describe the person's attributes in  the image in the following template:
                                                1. Height: You can guess the height or you can leave it blank
                                                2. Weight: You can guess the weight
                                                3. Skin color: Generalize the skin color of the person (for fashion recommendation)
                                                4. Shirt or top color:
                                                5. Pant or Bottoms color: If applicable
                                        """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + base64_image
                                }
                            }
                        ]
                    }
                    
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
                )
            
            st.session_state.photo_description[f"Image_{enu}"] = completion.choices[0].message
        st.session_state["uploader_key"] += 1
        st.rerun()

    # user_input = st.chat_input("What are you wearing?",)
    if prompt := st.chat_input("What are you wearing?", key = "initial_chat_input"):
        
    # --- INITIAL CONVERSATION ---

        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").markdown(prompt)

        if st.session_state.photo_description:
            prompt = prompt + "\n\n" + str(st.session_state.photo_description)
            st.session_state.photo_description = None

        if "first_render" not in st.session_state:
            st.session_state.first_render = True
            
            for update in  st.session_state.graph.stream({"messages": [("user", prompt)], "next": "summary"}, config= st.session_state.thread_config, stream_mode=["updates"]):

                # with st.chat_message("assistant"):
                #     st.markdown(update)
                # st.session_state.messages.append({"role": "assistant", "content": update})

                update_type, update_data = update

                agents = ['summary', 'dressing_style', 'color_recommender', 'outfit_recommender']

                for agent in agents:
                    content = ""
                    if agent in update_data:
                        print(f"response from {agent}")
                        agent_data = update_data[agent]
                        messages = agent_data.get("messages", [])
                        for msg in messages:
                            content += get_message_content(msg)
                        st.session_state.messages.append(AIMessage(content))
                        st.write(content)
                # st.session_state.checkpoint = checkpointer.get(thread_config)
            print(f"First Loop: { st.session_state.graph.get_state(config= st.session_state.thread_config)}\n")
        else:

            for resume_update in  st.session_state.graph.stream(Command(resume=prompt), config= st.session_state.thread_config, stream_mode=["updates"]):
                    # with st.chat_message("assistant"):
                    #     st.markdown(resume_update)
                    # st.session_state.messages.append({"role": "assistan t", "content": resume_update})

                    update_type, update_data = resume_update

                    agents = ['summary', 'dressing_style', 'color_recommender', 'outfit_recommender']

                    for agent in agents:
                        content = []
                        if agent in update_data:
                            print(f"2nd response from {agent}")
                            agent_data = update_data[agent]
                            messages = agent_data.get("messages", [])
                            for msg in messages:
                                content.append(get_message_content(msg))
                            st.session_state.messages.append(AIMessage(content[-1]))
                            st.write(content[-1])
                            # st.session_state.messages.append({"role": "assistant", "content": content})
            print(f"Second Loop: { st.session_state.graph.get_state(config= st.session_state.thread_config)}\n")
