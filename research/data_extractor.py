import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from llama_index.program.openai import OpenAIPydanticProgram
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

warnings.filterwarnings('ignore')
_ = load_dotenv()

parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="This is an invoice document that contains information about the patient, treatment, ",
    show_progress=True,
)

file_extractor = {".pdf": parser}

class Invoice(BaseModel):
    """Attributes of an invoice"""
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

def get_invoice_content(file_path: str) -> str:
    documents = SimpleDirectoryReader(input_files=[file_path],                      
                                  file_extractor=file_extractor).load_data()
    print(len(documents))
    return documents[0].get_content()

def get_invoice_data(invoice_path: str) -> Invoice:
    """
    Extract invoice data from the given text
    Args:
        invoice_text (str): The text content of the invoice    
    Returns:
        Invoice: Extracted invoice data in Invoice model format
    """
    prompt_template_str = """\
    You are an expert in extracting information from invoices.\
    Only extract invoice information from {invoice_text} in JSON and nothing else.\
    """
    invoice_text = get_invoice_content(invoice_path)

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=Invoice, 
        prompt_template_str=prompt_template_str, 
        verbose=True
    )
    
    return program(invoice_text=invoice_text)

# test
invoice_path = "./invoices/sample_invoice_1.pdf"
invoice_data = get_invoice_data(invoice_path)
print("invoice data", invoice_data)