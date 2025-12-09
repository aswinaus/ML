from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def classify_tax_text(text_chunk: str):
    """
    Classifies text as:
    - Tax Problem
    - Tax Solution
    - Neither
    """

    prompt = f"""
    You are a classifier. Read the following text and determine whether it contains:
    1. A TAX PROBLEM (customer question, issue, uncertainty)
    2. A TAX SOLUTION (an answer, guidance, or instructions)
    3. NEITHER (if it does not clearly express either)

    Respond ONLY in valid JSON using this schema:

    {{
       "classification": "Tax Problem" | "Tax Solution" | "Neither",
       "confidence": 0-100,
       "reason": "<short explanation>"
    }}

    Text to classify:
    \"\"\"{text_chunk}\"\"\"
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    # Extract text response
    result_text = response.output_text
    return result_text


# ------------------------------
# Example usage:
# ------------------------------

if __name__ == "__main__":
    sample_chunk = """
    The taxpayer is unsure whether they can claim a dependent exemption for
    their child who lived with them for only half the year.
    """

    result = classify_tax_text(sample_chunk)
    print("\nClassification Result:")
    print(result)
