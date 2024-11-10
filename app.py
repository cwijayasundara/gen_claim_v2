import streamlit as st
import os
import asyncio
from work_flow import create_workflow
from pathlib import Path
from invoice_data_extractor import extract_invoice_data
from prompts.claim_prompts import claim_processing_prompt, cash_back_prompt, final_response_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openinference.instrumentation.langchain import LangChainInstrumentor
import os
from phoenix.otel import register

_ = load_dotenv()

# set up Phoenix OTel instrumentation
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

# configure the Phoenix tracer
tracer_provider = register(
  project_name="default", # Default is 'default'
  endpoint="https://app.phoenix.arize.com/v1/traces",
  set_global_tracer_provider=False,
)

st.title("ClaimGenius : Your AI Assistant for Insurance Claims")

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

claim_processing_prompt_str = ChatPromptTemplate.from_template(claim_processing_prompt)

claim_chain = claim_processing_prompt_str | llm

cashback_prompt_str = ChatPromptTemplate.from_template(cash_back_prompt)

cash_back_chain = cashback_prompt_str | llm

final_response_prompt_str = ChatPromptTemplate.from_template(final_response_prompt)

final_response_chain = final_response_prompt_str | llm

async def invoke_wf(message):
    workflow = create_workflow()
    return await workflow.run(message=message)

def sanitize_filename(filename):
    """
    Sanitize the filename to prevent path traversal attacks and remove unwanted characters.
    """
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()
    return filename

def save_uploaded_file(uploaded_file):
    """Save the uploaded file and return the save path"""
    upload_dir = Path.cwd() / "uploaded_invoices"
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = Path(uploaded_file.name).name
    sanitized_filename = sanitize_filename(original_filename)
    save_path = str(upload_dir / sanitized_filename)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved successfully to: {save_path}")
    return save_path

async def get_policy_details(treatment_type):
    """Get policy details for the treatment type"""
    claim_section_prompt = f"What is the cashback amount for {treatment_type} fees per year?"
    return await invoke_wf(claim_section_prompt)

async def get_member_details(patient_name):
    """Get member details from the database"""
    member_data_extraction_prompt = f"""Get all the member details for {patient_name} from the database including last_claim_type, last_claim_amount, last_claim_date"""
    return await invoke_wf(member_data_extraction_prompt)

def process_claim(policy_section, claim_details, member_data):
    """Process the claim and return responses"""
    # Process initial claim
    claim_response = claim_chain.invoke({
        "POLICY_SECTION": policy_section,
        "CLAIM_DETAILS": claim_details,
        "MEMBER_DETAILS": member_data
    })
    
    # Get cashback amount
    cash_back_amount = cash_back_chain.invoke({"text": claim_response.content})
    
    return claim_response, cash_back_amount

def display_claim_info(policy_section, member_data, claim_details, claim_response, cash_back_amount, final_response):
    """Display all claim related information"""
    st.write(f"policy_section: :blue[{policy_section}]")
    st.markdown(f"member_data: <span style='color:green'>{member_data}</span>", unsafe_allow_html=True)
    st.markdown(f"claim_details_extracted: <span style='color:green'>{claim_details}</span>", unsafe_allow_html=True)
    st.write(f":blue[{claim_response.content}]")
    st.write(cash_back_amount.content)
    st.write(f":red[{final_response.content}]")

with st.sidebar:
    st.image("images/gen_claim.png", width=500)
    add_radio = st.radio(
        "What can I do for you today?",
        ("/chat with the knowledge base",
         "/make a claim!",
         "/show me the workflow!",
         "/show me the architecture!")
    )

if add_radio == "/chat with the knowledge base":
    st.header("/chat with the knowledge base")
    st.write("Example_1 : List all the member details for members with policy type Health?")
    st.write("Example-2: Whats the cashback amount for dental expenses?")
    st.write("Example-3: What's the accident that involved Michael Johnson?")
    request = st.text_area("", height=100)
    submit = st.button("submit", type="primary")

    if submit and request:
        result = asyncio.run(invoke_wf(request))
        st.write(result)

elif add_radio == "/make a claim!":
    st.header("/make a claim!")
    st.write("Please upload your invoice to get started")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file
        save_path = save_uploaded_file(uploaded_file)
        
        # Extract invoice data
        extracted_invoice_data = extract_invoice_data(save_path)
        st.write(extracted_invoice_data)
        
        # Get policy and member details
        policy_section = asyncio.run(get_policy_details(extracted_invoice_data.treatment_type))
        member_data = asyncio.run(get_member_details(extracted_invoice_data.patient_name))
        
        # Prepare claim details
        claim_details = f"{extracted_invoice_data.invoice_total} {extracted_invoice_data.treatment_type}"
        
        # Process the claim
        claim_response, cash_back_amount = process_claim(policy_section, claim_details, member_data)
        
        # Generate final response
        final_response = final_response_chain.invoke({
            "invoice_data": extracted_invoice_data,
            "claim_decision": claim_response.content
        })
        
        # Display all information
        display_claim_info(
            policy_section,
            member_data,
            claim_details,
            claim_response,
            cash_back_amount,
            final_response
        )

elif add_radio == "/show me the workflow!":
    st.image("images/wf.png", width=800)

elif add_radio == "/show me the architecture!":
    st.image("images/arch.png", width=800)