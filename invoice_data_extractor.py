import warnings
import ssl
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import UnstructuredPDFLoader

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Invoice(BaseModel):
    """Information about the items on an invoice"""
    invoice_number:str = Field(description="The invoice number on the invoice. Example: 2112407002889")
    invoice_date:str = Field(description="The issue date on the invoice. Format: DD/MM/YYYY HH:MM. Example: "
                                         "09/07/2024 12:47")
    patient_name:str = Field(description="The patient name given on the invoice. Example: John Doe")
    patient_address:str = Field(description="The patient address given on the invoice. This is a standard UK address "
                                            "format. If you can't find the address, just output 'Not Found'.")
    treatment_type:str = Field(description="The type of treatment given to the patient. dental, optical, etc.")
    invoice_total:str = Field(description="The total of the invoice that the patient has to pay")
    clinic_name:str = Field(description="The name of the clinic that issued the invoice. Example: Dental Clinic. "
                                        "If you can't find the clinic name, just output 'Not Found")
    clinic_address:str = Field(description="The address of the clinic that issued the invoice. This is a standard UK "
                                           "address. If you can't find the address, just output 'Not Found'")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in extracting information from invoices. "
            "Only extract invoice information in JSON and nothing else."
        ),
        ("human", "{text}"),
    ]
)

structured_llm = llm.with_structured_output(Invoice)

def get_invoice_content(file_path):
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    return data[0].page_content

def extract_invoice_data_by_str(invoice_content: str) -> dict:
    response = structured_llm.invoke(invoice_content)
    return response

#  write a function to extract the invoice data
def extract_invoice_data(invoice_location_str: str) -> dict:
    invoice_content = get_invoice_content(invoice_location_str)
    response = structured_llm.invoke(invoice_content)
    return response

# test the invoice data extraction function
# invoice_location = "./invoices/sample_invoice_1.pdf"
# invoice_data = extract_invoice_data(invoice_location)
# print(invoice_data)

# Option 1: Access directly using dot notation
# invoice_number = invoice_data.invoice_number
# print(f"Invoice Number: {invoice_number}")

# Option 2: Convert to dictionary first if you need dictionary access
# invoice_dict = invoice_data.model_dump()
# invoice_number = invoice_dict["invoice_number"]
# print(f"Invoice Number: {invoice_number}")