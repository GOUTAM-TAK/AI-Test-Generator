import traceback
import gradio as gr
import openai
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import os
import uuid
import json
import logging
import ast
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Setup logging with detailed format and multiple handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
#env = envs()
load_dotenv()

# Initialize OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure you set this environment variable
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
# Path to your ChromeDriver
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")

client =  OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
def extract_functionalities(url):
    """
    Extract functionalities from the given URL.
    """
    driver = None  # Initialize driver variable outside try block
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Optional: Run in headless mode
        service = ChromeService(executable_path=CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))  # Wait until the body tag is present

        # Explicitly retrieve the page source and ensure UTF-8 encoding
        page_source = driver.page_source.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        functionalities = []

        # Extract forms (e.g., login, signup)
        forms = driver.find_elements(By.TAG_NAME, 'form')
        for form in forms:
            form_id = form.get_attribute('id') or form.get_attribute('name') or 'form'
            functionalities.append({'type': 'form', 'identifier': form_id})

        # Extract buttons
        buttons = driver.find_elements(By.TAG_NAME, 'button')
        for button in buttons:
            btn_text = button.text.strip()
            if btn_text:
                functionalities.append({'type': 'button', 'identifier': btn_text})

        # Extract links
        links = driver.find_elements(By.TAG_NAME, 'a')
        for link in links:
            link_text = link.text.strip()
            href = link.get_attribute('href')
            if link_text and href:
                functionalities.append({'type': 'link', 'identifier': link_text, 'href': href})

        # Remove duplicates
        unique_funcs = [dict(t) for t in {tuple(d.items()) for d in functionalities}]
        logging.info(f"Extracted functionalities: {unique_funcs}")
        print("Extracted functionalities:",unique_funcs)
        return unique_funcs

    except Exception as e:
        logging.error(f"Error in extract_functionalities: {e}")
        logging.error(traceback.format_exc())  # Log the stack trace for better debugging
        return {"error": str(e)}

    finally:
        # Ensure the driver quits even if there's an error
        if driver:
            driver.quit()

def generate_test_case(functionality, url):
    """
    Generate a Selenium-compatible test case for the given functionality using OpenAI's GPT.
    """
    try:
        prompt = f"""
        Given the following website URL: {url}

        Generate a Selenium WebDriver Python function named `test_{functionality['type']}_{re.sub(r'\\s+', '_', functionality['identifier'])}` that tests the specified functionality.

        Functionality Type: {functionality['type']}
        Identifier: {functionality['identifier']}

        The test should be functional and non-functional where applicable.
        Provide only the Python code using Selenium, without any markdown formatting, comments, or code block delimiters.
        Ensure proper indentation and syntax.
        """

        # Make the API call using the `openai` client
        completion = client.chat.completions.create(
            model='abacusai/dracarys-llama-3.1-70b-instruct',
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Selenium test cases."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        # Stream and print each chunk of response
        code = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                code_chunk = chunk.choices[0].delta.content
                print(code_chunk, end="")
                code += code_chunk

        return code

    except Exception as e:
        logging.error(f"Error generating test case: {e}")
        print("error in generating test cases : ",e)
        return f"# Error generating test case: {str(e)}"

def sanitize_code(code):
    """
    Cleans the generated code by removing any markdown code block delimiters and language specifiers.
    """
    # Remove code block delimiters if present
    code = re.sub(r'^```python', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```', '', code, flags=re.MULTILINE)
    code = re.sub(r'```$', '', code, flags=re.MULTILINE)
    
    # Remove any language specifiers like 'python' at the start of a line
    code = re.sub(r'^\s*python\s*', '', code, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    return code

def is_valid_python(code):
    """
    Checks if the provided code is valid Python syntax.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logging.error(f"Syntax error in generated code: {e}")
        return False

def execute_test_cases(test_cases, url):
    """
    Execute the generated test cases using Selenium and collect the results.
    """
    results = []
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Optional: Run in headless mode
        service = ChromeService(executable_path=CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))  # Wait until the body tag is present

        for idx, test_case in enumerate(test_cases):
            test_id = test_case['id']
            code = sanitize_code(test_case['code'])
            logging.info(f"Executing Test ID: {test_id}\nCode:\n{code}\n{'-'*60}")
            
            if not is_valid_python(code):
                results.append({"test_id": test_id, "status": "Failed", "details": "Invalid Python syntax in generated code."})
                continue

            try:
                # Define a local scope for exec
                local_scope = {'driver': driver, 'By': By, 'time': time}
                exec(code, {}, local_scope)
                results.append({"test_id": test_id, "status": "Passed", "details": "Test executed successfully."})
            except Exception as e:
                logging.error(f"Error executing test {test_id}: {e}")
                results.append({"test_id": test_id, "status": "Failed", "details": str(e)})

        driver.quit()
    except Exception as e:
        logging.error(f"Error in execute_test_cases: {e}")
        for test_case in test_cases:
            results.append({"test_id": test_case['id'], "status": "Failed", "details": str(e)})
    return results

def ai_test_generator(url):
    """
    Main function to handle the AI test generation process.
    """
    if not re.match(r'^https?:\/\/', url):
        url = 'https://' + url

    extraction = extract_functionalities(url)
    if isinstance(extraction, dict) and "error" in extraction:
        return f"Error extracting functionalities: {extraction['error']}"

    functionalities = extraction
    if not functionalities:
        return "No functionalities found on the provided URL."

    test_cases = []
    for func in functionalities:
        code = generate_test_case(func, url)
        test_id = str(uuid.uuid4())[:8]
        test_cases.append({"id": test_id, "functionality": func, "code": code})

    # Execute test cases
    execution_results = execute_test_cases(test_cases, url)

    # Prepare results in a vertical format
    results_list = []
    for test, result in zip(test_cases, execution_results):
        functionality = test['functionality']
        functionality_str = (
            f"Type: {functionality['type']}\n"
            f"Identifier: {functionality['identifier']}\n"
            f"Href: {functionality.get('href', 'N/A')}\n"
        )

        results_list.append({
            "Test ID": test['id'],
            "Functionality": functionality_str,
            "Result": result['status']
        })

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results_list)
    return results_df

# Gradio Interface using updated components
iface = gr.Interface(     #Interface is class
    fn=ai_test_generator,
    inputs=gr.Textbox(lines=2, placeholder="Enter the URL of the website to test...", label="Website URL"),
    outputs=gr.Dataframe(label="Test Cases and Results"),  # Change from Textbox to Dataframe
    title="AI Test Generator",
    description="Enter a website URL to automatically generate and execute Selenium test cases based on the site's functionalities.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()