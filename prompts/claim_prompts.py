claim_processing_prompt = f"""

You are an AI assistant tasked with generating a prompt to process health insurance claims. 
You will be provided with three key pieces of information: 
- a section from the insurance policy that defines the claim category and its associated guidelines
- details extracted from the claim invoice
- members details from the database including last_claim_type, last_claim_amount, last_claim_date
Your task is to analyze this information and generate a prompt that will guide the processing of the health 
insurance claim.

First, carefully read and analyze the following section from the insurance policy:

<policy_section>
{{POLICY_SECTION}}
</policy_section>

Next, review the details extracted from the claim invoice:

<claim_details>
{{CLAIM_DETAILS}}
</claim_details>

After that, review the member details extracted from the database:

<member_details>
{{MEMBER_DETAILS}}
</member_details>

Now, follow these steps to generate the prompt:

1. Analyze the policy section:
   - Identify the claim category (e.g., dental, optical, outpatient)
   - Note the maximum claimable amount or percentage
   - List any specific conditions or limitations mentioned

2. Compare the claim details with the policy guidelines:
   - Check if the claimed category matches the policy section
   - Verify if the claim amount is within the policy limits
   - Identify any relevant details that need to be considered based on the policy conditions
   - calculate the total claim amount based on the policy guidelines

3. Check the member details:
   - Review the member's last claim type, amount, and date. The total claim amount should not exceed the maximum claimable amount
   - Consider the member's history for the current claim processing
   - Identify any specific details that may impact the claim processing

4. Generate a prompt for processing the claim:
   - Start with a clear instruction to process the claim
   - Include relevant details from the policy and claim
   - Formulate specific questions to guide the claim processing

5. Format your output as follows:
   <decision>
   [Provide a brief explanation of how you arrived at this prompt, referencing specific parts of the policy 
   and claim details and member claim history with the calculated claim amount]
   </decision>
   <cashback_amount>
    [Provide the calculated claim amount based on the policy guidelines and member claim history]
   </cashback_amount>

Remember to make the prompt clear, concise, and specific to the given policy, member claim history and claim details. 
The prompt should guide the claim processor to make an accurate decision based on the provided information.

"""

cash_back_prompt = """

Can you extract the cashback amount and the treatment type from this text?

{text}

Produce the extracted cashback amount and treatment type with the the below attributes in a json format:

treatment_type, cashback_amount

"""

final_response_prompt = """

You are tasked with writing a professional email to an insurance claiming client, informing them of the final decision 
on their claim and the final claim amount. You will be provided with invoice data and the claim decision. Your goal is 
to compose a clear, concise, and professional email that effectively communicates the outcome of the claim.

Here is the invoice data:
<invoice_data>
    {invoice_data}
</invoice_data>

Here is the claim decision:

<claim_decision>
    {claim_decision}
</claim_decision>

Using the provided information, write an email to the client following these guidelines:

1. Start with a professional greeting.
2. In the opening paragraph, briefly acknowledge the client's claim and mention the claim number if available.
3. Clearly state the final decision on the claim (approved, partially approved, or denied).
4. If the claim is approved or partially approved, specify the final claim amount.
5. Provide a brief explanation for the decision, referencing relevant details from the invoice data and claim decision.
6. If applicable, mention any next steps the client needs to take or when they can expect to receive the payment.
7. Close the email with a professional sign-off and your name/title.

Maintain a professional and empathetic tone throughout the email. Be clear and concise in your communication, avoiding 
technical jargon where possible. If the claim is denied or partially approved, be tactful in explaining the reasons 
while remaining firm on the decision.

Write your email inside <email> tags. Do not include any commentary or explanations outside of these tags.

"""