import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
import re
import base64
import openai

# --- Helper Functions from n8n workflow ---

def extract_logo_and_text(html, url):
    """
    Extracts logo, text, and title from HTML content.
    This is a Python implementation of the 'Extract Logo & Text' node.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # --- LOGO Extraction ---
        logo = ''
        logo_selectors = [
            'img[src*="logo"]', 'img[alt*="logo"]', 'img[class*="logo"]', 'img[id*="logo"]',
            'svg[class*="logo"]', 'svg[id*="logo"]'
        ]
        for selector in logo_selectors:
            logo_tag = soup.select_one(selector)
            if logo_tag:
                if logo_tag.name == 'img':
                    logo = logo_tag.get('src')
                elif logo_tag.name == 'svg':
                    logo = str(logo_tag) # Return the whole SVG tag
                if logo:
                    break
        
        if not logo:
            # Fallback to favicon
            favicon_selectors = [
                'link[rel="icon"]', 'link[rel="shortcut icon"]', 'link[rel="apple-touch-icon"]'
            ]
            for selector in favicon_selectors:
                favicon_tag = soup.select_one(selector)
                if favicon_tag:
                    logo = favicon_tag.get('href')
                    if logo:
                        break

        # --- TEXT Extraction ---
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            tag.decompose()

        main_content = soup.select_one('main')
        if not main_content:
            main_content = soup.select_one('#content')
        if not main_content:
            main_content = soup.select_one('.content')
        if not main_content:
            main_content = soup.body

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:8000] # Increased limit for better context

        # --- TITLE Extraction ---
        title = soup.title.string if soup.title else ''

        return {
            "url": url,
            "logo": logo,
            "text": text,
            "title": title,
            "error": False
        }
    except Exception as e:
        return {
            "url": url,
            "logo": '',
            "text": f'Error processing webpage: {e}',
            "title": '',
            "error": True
        }


def get_placeholders(template_content):
    """Extracts all unique placeholders (e.g., %placeholder%) from the template."""
    return set(re.findall(r"%([a-zA-Z0-9_]+)%", template_content))

def create_dynamic_schema(placeholders):
    """Creates a nested dictionary schema from a list of flattened placeholder keys."""
    schema = {}
    for key in placeholders:
        parts = key.split('_')
        d = schema
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = "" # Set a dummy value
    return schema

def call_ai_agent(prompt, text, url, logo, title, api_key, model, dynamic_schema):
    """
    Calls the OpenRouter API to generate content based on a dynamic schema.
    """
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Instruct the AI to only fill the fields present in the dynamic schema
    schema_prompt = json.dumps(dynamic_schema, indent=2)
    full_prompt = f"{prompt}\n\nHere is the content of my landing page: {text}\nUse also this information about the website: URL: {url}, Title: {title}, Logo: {logo}\n\nBased on the information, please populate the following JSON structure. It is crucial that you fill every field. If specific information is not available on the page, use your knowledge to generate a sensible and high-quality default value appropriate for the business type. Do not leave any fields blank or empty.\n\n{schema_prompt}\n\nRespond ONLY with a valid JSON object that matches the requested schema."

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                },
            ],
            response_format={"type": "json_object"},
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        st.error(f"Error calling AI Agent: {e}")
        return {
            "header": {"logo": logo, "nav_menu": {}},
            "hero_section": {"headline": f"Error generating headline for {title}"},
            "about_section": {},
            "services_section": {},
            "mission_section": {},
            "stats_section": {},
            "testimonials_section": {},
            "cta_final_section": {},
            "footer": {}
        }


def flatten_json(y):
    """Flattens a nested JSON."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def deep_merge(source, destination):
    """
    Deeply merges source dictionary into destination dictionary.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value

    return destination

def replace_placeholders(html_template, data):
    """Replaces placeholders in the HTML template with data."""
    flat_data = flatten_json(data)
    for key, value in flat_data.items():
        placeholder = f"%{key}%"
        if value: # Only replace if value is not empty
            if key == 'header_logo' or key == 'footer_logo':
                if value.strip().startswith('<svg'):
                    html_template = html_template.replace(placeholder, value)
                else:
                    html_template = html_template.replace(placeholder, f'<img src="{value}" alt="Logo" style="max-width: 150px; height: auto;">')
            else:
                html_template = html_template.replace(placeholder, str(value))
    # Clean up any remaining placeholders
    html_template = re.sub(r"%([a-zA-Z0-9_]+)%", "", html_template)
    return html_template

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="AI Website Content Generator", page_icon="✨")

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        border: 1px solid #E6E6E6;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: white;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1rem;
    }
    h1 {
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ AI Website Content Generator")
st.write("This tool uses AI to generate website content based on a URL and an HTML template.")


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# --- App Configuration ---
openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
selected_model = st.secrets["DEFAULT_MODEL"]

st.sidebar.success("Configuration loaded securely.")
st.sidebar.info(f"Using model: `{selected_model}`")


# --- Main Content Area ---
st.header("1. Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a CSV with a 'URLs' column", type="csv")

with col2:
    # Load the HTML template automatically
    try:
        with open("templates/template.html", "r", encoding="utf-8") as f:
            html_template = f.read()
        st.success("`template.html` loaded successfully.")
    except FileNotFoundError:
        st.error("`templates/template.html` not found. Please create it.")
        st.stop()


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'URLs' in df.columns:
            st.header("2. Processing and Results")
            total_urls = len(df['URLs'])
            st.info(f"Found {total_urls} URL(s) to process.")
            
            results = []
            
            with st.spinner('Processing URLs... This may take a moment.'):
                for i, url in enumerate(df['URLs']):
                    try:
                        # Ensure URL has a scheme
                        if not url.startswith(('http://', 'https://')):
                            url = 'https://' + url

                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        
                        html_content = response.text
                        
                        # 1. Extract Logo and Text
                        extracted_data = extract_logo_and_text(html_content, url)
                        if extracted_data["error"]:
                            st.error(f"Could not process {url}: {extracted_data['text']}")
                            continue

                        # 2. AI Agent Analysis
                        placeholders = get_placeholders(html_template)
                        dynamic_schema = create_dynamic_schema(placeholders)

                        ai_prompt_1 = "You are an expert web page analyzer. Your task is to extract key information from the provided web page content and populate a JSON object. Focus on understanding the business, its services, and its mission."
                        
                        generated_content_1 = call_ai_agent(
                            ai_prompt_1,
                            extracted_data["text"],
                            url,
                            extracted_data["logo"],
                            extracted_data["title"],
                            openrouter_api_key,
                            selected_model,
                            dynamic_schema
                        )

                        ai_prompt_2 = "You are an expert marketing analyst. Your task is to generate the missing information based on what you have from the given website, and what you know is going to work for the new website. Fill in any empty fields in the provided JSON object to create a complete, marketing-focused content set."
                        
                        prompt_2_with_context = f"{ai_prompt_2}\n\nHere is the partially filled JSON from the first analysis:\n{json.dumps(generated_content_1, indent=2)}"

                        generated_content_2 = call_ai_agent(
                            prompt_2_with_context,
                            extracted_data["text"],
                            url,
                            extracted_data["logo"],
                            extracted_data["title"],
                            openrouter_api_key,
                            selected_model,
                            dynamic_schema
                        )
                        
                        generated_content = deep_merge(generated_content_1, generated_content_2)
                        
                        final_html = replace_placeholders(html_template, generated_content)
                        
                        try:
                            from urllib.parse import urlparse
                            parsed_url = urlparse(url)
                            domain = parsed_url.netloc.replace('www.', '')
                            file_name_base = domain.split('.')[0]
                            file_name = f"{file_name_base}.html"
                        except:
                            file_name = "download.html"

                        results.append({
                            'URL': url,
                            'GeneratedHTML': final_html,
                            'FileName': file_name,
                            'Logo': extracted_data.get('logo', ''),
                            'Title': extracted_data.get('title', ''),
                        })

                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to fetch {url}: {e}")
                    except Exception as e:
                        st.error(f"An error occurred while processing {url}: {e}")

            if results:
                st.success("Processing complete!")
                st.header("3. Download Your Files")
                
                result_df = pd.DataFrame(results)

                for index, row in result_df.iterrows():
                    with st.expander(f"📁 {row['URL']}"):
                        st.download_button(
                            label=f"Download {row['FileName']}",
                            data=row['GeneratedHTML'],
                            file_name=row['FileName'],
                            mime='text/html',
                        )
                        with st.container():
                            st.code(row['GeneratedHTML'], language='html')
                
                # Provide a separate download for the summary CSV
                csv_summary = result_df[['URL', 'Title', 'Logo']].to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Download Summary CSV",
                    data=csv_summary,
                    file_name='summary.csv',
                    mime='text/csv',
                )

        else:
            st.error("The uploaded CSV file must contain a 'URLs' column.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
