import streamlit as st
import feedparser
import json
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from openai import OpenAI
import anthropic
import config


# Initialize session state for LLM selection
if 'llm' not in st.session_state:
    st.session_state.llm = "anthropic"

# Initialize session state for LLM selection
if 'summary_length' not in st.session_state:
    st.session_state.summary_length = "paragraph"

# LLM Selection
st.sidebar.title("Configuration")
st.sidebar.subheader("Select LLM Service")
llm_option = st.sidebar.selectbox(
    "LLM Service",
    ("openai", "anthropic", "llama3", "mistral"),
    index=["openai", "anthropic", "llama3", "mistral"].index(st.session_state.llm)
)

# Update LLM in session state
if llm_option != st.session_state.llm:
    st.session_state.llm = llm_option
    st.experimental_rerun()

# Use the LLM from session state
llm = st.session_state.llm

st.sidebar.subheader("Long vs. Short Summary")
summary_option = st.sidebar.selectbox(
    "Summary Length",
    ("paragraph", "oneliner"),
    index=["paragraph", "oneliner"].index(st.session_state.summary_length)
)

# Update LLM in session state
if summary_option != st.session_state.summary_length:
    st.session_state.summary_length = summary_option
    st.experimental_rerun()

# Use the ummary length from session state
summary_length = st.session_state.summary_length

if llm == "openai":
    # openai_api_key = ''
    client = OpenAI(api_key=openai_api_key)
elif llm == "anthropic":
    # claude_api_key = ''
    client = anthropic.Anthropic(api_key=claude_api_key)
else:
    # else > llama3 or mistral, both via HF endpoint
    # hf_api_key = ''

# File paths for the JSON files
FEED_FILE = 'feeds.json'
DELETED_ENTRIES_FILE = 'deleted_entries.json'

# Initialize session state
if 'feeds' not in st.session_state:
    st.session_state.feeds = []
if 'feed_data' not in st.session_state:
    st.session_state.feed_data = {}
if 'read_items' not in st.session_state:
    st.session_state.read_items = set()
if 'deleted_entries' not in st.session_state:
    st.session_state.deleted_entries = set()
if 'scraped_content' not in st.session_state:
    st.session_state.scraped_content = {}
if 'summaries' not in st.session_state:
    st.session_state.summaries = []
if 'marked_for_scraping' not in st.session_state:
    st.session_state.marked_for_scraping = []
if 'manual_urls' not in st.session_state:
    st.session_state.manual_urls = []

# UTILITY FUNCTIONS

def load_feeds():
    if os.path.exists(FEED_FILE):
        with open(FEED_FILE, 'r') as file:
            data = json.load(file)
            st.session_state.feeds = data.get('feeds', [])

def save_feeds():
    with open(FEED_FILE, 'w') as file:
        json.dump({'feeds': st.session_state.feeds}, file, indent=4)

def load_deleted_entries():
    if os.path.exists(DELETED_ENTRIES_FILE):
        with open(DELETED_ENTRIES_FILE, 'r') as file:
            data = json.load(file)
            st.session_state.deleted_entries = set(data.get('deleted_entries', []))

def save_deleted_entries():
    with open(DELETED_ENTRIES_FILE, 'w') as file:
        json.dump({'deleted_entries': list(st.session_state.deleted_entries)}, file, indent=4)

def fetch_feed_data():
    feed_data = {}
    for feed in st.session_state.feeds:
        url = feed['url']
        source = feed['source']
        parsed_feed = feedparser.parse(url)
        if parsed_feed.bozo:
            st.error(f"Failed to parse feed: {url}")
            continue
        feed_data[source] = []
        for entry in parsed_feed.entries:
            item = {
                'title': entry.title,
                'link': entry.link,
                'published': entry.published if 'published' in entry else 'N/A',
                'summary': entry.summary if 'summary' in entry else 'N/A',
            }
            feed_data[source].append(item)
    st.session_state.feed_data = feed_data

def delete_entry(item_link):
    st.session_state.deleted_entries.add(item_link)
    save_deleted_entries()
    st.experimental_rerun()  # Immediately rerun the script to update the UI

def filter_feed_items(search_query):
    filtered_data = {}
    for source, items in st.session_state.feed_data.items():
        filtered_data[source] = [
            item for item in items
            if search_query.lower() in item['title'].lower() and item['link'] not in st.session_state.deleted_entries
        ]
    return filtered_data

def add_feed(source, url):
    if not any(feed['url'] == url for feed in st.session_state.feeds):
        st.session_state.feeds.append({'source': source, 'url': url})
        fetch_feed_data()
        save_feeds()

def remove_feed(url):
    st.session_state.feeds = [feed for feed in st.session_state.feeds if feed['url'] != url]
    fetch_feed_data()
    save_feeds()

def scrape_article_with_selenium(url):
    try:
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        # Extract text content from common tags
        article_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'div', 'article', 'section'])])
        return article_text
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return "Error: Could not scrape the article."

def summarize_article(content, entry):
    SYSTEM_PROMPT = "You are a helpful assistant that briefly and concisely summarizes news articles."
    if summary_length == "paragraph":
        USER_PROMPT = (
            "STEP 1: "
            "Summarize the following news article in two or three short sentences in German and provide three versions of "
            "a meaningful German headline for the summary. "
            " The format is: "
            " 1. first headline "
            " 2. second headline "
            " 3. third headline "
            " summary text...\n"
            "Do not translate technical terms like Large Language Model, Machine Learning, Artificial General Intelligence "
            "(AGI) etc. which are commonly used also in German text. "
            "The text might include additional garbage from the web page like menu entries or ads, ignore that, "
            "just use the text that obviously is part of the news article.  "
            "Do not use expressions like 'German Headline' or 'Zusammenfassung' or 'Die Zusammenfassung in zwei Sätzen' "
            "or 'A two-sentence summary of...' or anything like that. Just output headlines and the summary text. "
            "And keep the summary short, only two or three short sentences.\n"
            "STEP 2:"
            "Then, as a second step, also summarize the following news article in one very short sentence in German. "
            "Prefix the line with three suitables emojis."
            "Examples:\n"
            "💪🧑‍🤝‍🧑😁 Neue McKinsey-Studie: Agenten-Netzwerke vervielfachen die Leistung von LLMs.\n"
            "🤑💰💻 nvidias Marktkapitalisierung steigt auf über 3 Billionen US-$.\n"
            "🎥🎬🍿 OpenAIs Sora bringt die Filmbranche durcheinander.\n"
            "Keep it short, activating, teasing, engaging. The goal is that people click on the text because it sounds so interesting!"
            "\n\nContext:\n"
        )
    else:
        USER_PROMPT = (
            "Summarize the following news article in one very short sentence in German. Prefix the line with three suitables emojis."
            "Examples:\n"
            "💪🧑‍🤝‍🧑😁 Neue McKinsey-Studie: Agenten-Netzwerke vervielfachen die Leistung von LLMs.\n"
            "🤑💰💻 nvidias Marktkapitalisierung steigt auf über 3 Billionen US-$.\n"
            "🎥🎬🍿 OpenAIs Sora bringt die Filmbranche durcheinander.\n"
            "Keep it short, activating, teasing, engaging. The goal is that people click on the text because it sounds so interesting!"
            "\n\nContext:\n"
        )

    try:
        if llm == "openai":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + content}
            ]
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                max_tokens=350  # Increased max tokens to account for both headline and summary
            )
            response_text = response.choices[0].message.content.strip()
        elif llm == "anthropic":
            message = client.messages.create(
                #model="claude-3-sonnet-20240229",
                model="claude-3-5-sonnet-20240620",
                max_tokens=350,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT + content}
                ]
            )
            # Make the API call to Anthropic's Claude 3 Sonnet
            response_text = message.content[0].text
        elif llm == "llama3":
            # model_name = "meta-llama/Meta-Llama-3-70B-instruct"
            model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k, see endpoint below"
            #model_name = "inflaton/Llama-3-8B-Instruct-Gradient-1048k-MAC-lora"
            #model_name = "leafspark/Llama-3-8B-Instruct-Gradient-4194k-GGUF"
            prompt = (
                    "system\n" + SYSTEM_PROMPT +
                     "\nuser\n" + USER_PROMPT +
                    "\n\nContext: " + content +
                    "assistant"
            )
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {
                    "inputs": prompt,
                    "parameters": {
                        "min_length": 20,
                        "max_new_tokens": 350,
                        "repetition_penalty": 1.18,
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_gpu": True
                    }
            }
            response = requests.post(
                #f"https://api-inference.huggingface.co/models/{model_name}",
                #"https://jaa1089lcyfpl8lx.us-east-1.aws.endpoints.huggingface.cloud", # 70B
                "https://uaqz6uo5pmqzlczb.us-east-1.aws.endpoints.huggingface.cloud", # 8B
                headers=headers,
                json=payload
            )
            response_json = response.json()
            response_text = json.dumps(response_json, indent=4)
            st.write(response_text)
            #response_text = response_json[0]["generated_text"][len(prompt):]
        else:
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            prompt = "[INST]" + SYSTEM_PROMPT + "[/INST]\n\n" + USER_PROMPT + "\n\n" + content
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "min_length": 20,
                    "max_new_tokens": 350,
                    "repetition_penalty": 1.18,
                },
            }

            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers=headers,
                json=payload
            )
            response_json = response.json()
            #response_text = json.dumps(response_json, indent=4)
            response_text = response_json[0]["generated_text"][len(prompt):]

        st.session_state.summaries.insert(0, {
            'headline': "",
            'summary': response_text,
            'url': entry['link']
        })
    except Exception as e:
        st.error(f"An error occurred: {e}")

def summarize_article2(content, entry):
    SYSTEM_PROMPT = "You are a helpful assistant that briefly and concisely summarizes news articles."
    if summary_length == "paragraph":
        USER_PROMPT = (
            "Summarize the following news article in two or three short sentences in German and provide three versions of "
            "a meaningful German headline for the summary. "
            " The format is: "
            " 1. first headline "
            " 2. second headline "
            " 3. third headline "
            " summary text...\n"
            "Do not translate technical terms like Large Language Model, Machine Learning, Artificial General Intelligence "
            "(AGI) etc. which are commonly used also in German text. "
            "The text might include additional garbage from the web page like menu entries or ads, ignore that, "
            "just use the text that obviously is part of the news article.  "
            "Do not use expressions like 'German Headline' or 'Zusammenfassung' or 'Die Zusammenfassung in zwei Sätzen' "
            "or 'A two-sentence summary of...' or anything like that. Just output headlines, summary text and hashtags. "
            "And keep the summary short, only two or three short sentences."
            "\n\nContext:\n"
        )
    else:
        USER_PROMPT = (
            "Summarize the following news article in one very short sentence in German. Prefix the line with three suitables emojis."
            "Examples:\n"
            "💪🧑‍🤝‍🧑😁 Neue McKinsey-Studie: Agenten-Netzwerke vervielfachen die Leistung von LLMs.\n"
            "🤑💰💻 nvidias Marktkspitlisierung steigt auf über 3 Billiarden US-$.\n"
            "🎥🎬🍿 OpenAIs Sora bringt die Filmbranche durcheinander.\n"
            "Keep it short, activating, teasing, engaging. The goal is that people click on the text because it sounds so interesting!"
            "\n\nContext:\n"
        )

    try:
        if llm == "openai":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + content}
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=350  # Increased max tokens to account for both headline and summary
            )
            response_text = response.choices[0].message.content.strip()
        elif llm == "anthropic":
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=350,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT + content}
                ]
            )
            # Make the API call to Anthropic's Claude 3 Sonnet
            response_text = message.content[0].text
        elif llm == "llama3":
            #model_name = "meta-llama/Meta-Llama-3-70B-instruct"
            model_name = "gradientai/Llama-3-8B-Instruct-262k"
            prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + SYSTEM_PROMPT +
                     "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + USER_PROMPT +
                    "\n\nContext: " + content +
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {
                    "inputs": prompt,
                    "parameters": {
                        "min_length": 20,
                        "max_new_tokens": 350,
                        "repetition_penalty": 1.18,
                    },
                    "options": {
                    "wait_for_model": True,
                    "use_gpu": True
                    }
            }
            response = requests.post(
                  f"https://api-inference.huggingface.co/models/{model_name}",
                  headers=headers,
                  json=payload
            )
            response_json = response.json()
            response_text = json.dumps(response_json, indent=4)
            st.write(response_text)
            #response_text = response_json[0]["generated_text"][len(prompt):]
        else:
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            prompt = "[INST]" + SYSTEM_PROMPT + "[/INST]\n\n" + USER_PROMPT + "\n\n" + content
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "min_length": 20,
                    "max_new_tokens": 350,
                    "repetition_penalty": 1.18,
                },
            }

            response = requests.post(
              f"https://api-inference.huggingface.co/models/{model_name}",
              headers=headers,
                json=payload
            )
            response_json = response.json()
            #response_text = json.dumps(response_json, indent=4)
            response_text = response_json[0]["generated_text"][len(prompt):]

        st.session_state.summaries.insert(0, {
            'headline': "",
            'summary': response_text,
            'url': entry['link']
        })
        #st.experimental_rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def delete_summary(index):
    del st.session_state.summaries[index]
    st.experimental_rerun()

def mark_for_scraping(item_link):
    if item_link not in st.session_state.marked_for_scraping:
        st.session_state.marked_for_scraping.append(item_link)
        st.session_state[f"marked_{item_link}"] = True

def process_marked_entries():
    st.session_state.temp_marked_for_scraping = st.session_state.marked_for_scraping.copy()
    for link in st.session_state.temp_marked_for_scraping:
        try:
            if link in st.session_state.scraped_content:
                content = st.session_state.scraped_content[link]
            else:
                content = scrape_article_with_selenium(link)
                st.session_state.scraped_content[link] = content

            # Find the corresponding entry for summarizing
            entry_found = False
            for source, items in st.session_state.feed_data.items():
                for item in items:
                    if item['link'] == link:
                        summarize_article(content, item)
                        entry_found = True
                        break
                if entry_found:
                    break

            # If no entry was found in feed data, handle it as a manual URL
            if not entry_found:
                summarize_article(content, {'link': link})

            # Remove processed link from marked entries
            st.session_state.marked_for_scraping.remove(link)
        except Exception as e:
            st.error(f"An error occurred while processing {link}: {e}")
            # Restore the marked entries
            st.session_state.marked_for_scraping = st.session_state.temp_marked_for_scraping.copy()
            break

    # Clear the temp marked entries after processing
    if 'temp_marked_for_scraping' in st.session_state:
        del st.session_state.temp_marked_for_scraping
    # st.session_state.marked_for_scraping.clear()

st.button("Process Marked Entries", on_click=process_marked_entries)

if not st.session_state.feeds:
    load_feeds()
    fetch_feed_data()
if not st.session_state.deleted_entries:
    load_deleted_entries()


# USER INTERFACE
# Load feeds and deleted entries from the JSON files on startup

st.markdown('<a name="top"></a>', unsafe_allow_html=True)
st.title("RSS Feed Reader")

# Create anchor links for each feed
st.sidebar.header("Jump to Feed")
st.sidebar.markdown("[Top](#top)")
for source in st.session_state.feed_data.keys():
    st.sidebar.markdown(f"[{source}](#{source.replace(' ', '-').lower()})")

with st.sidebar:
    # Manual URL input
    st.header("Add URLs Manually")
    manual_urls = st.text_input("Enter URLs (comma-separated)")
    if st.button("Add URLs"):
        if manual_urls:
            urls = [url.strip() for url in manual_urls.split(",") if url.strip()]
            for url in urls:
                if url:
                    st.session_state.manual_urls.append(url)
                    mark_for_scraping(url)
            st.success(f"URLs added and marked for scraping.")

    st.subheader("Your Feeds")
    for feed in st.session_state.feeds:
        st.text(f"{feed['source']} - {feed['url']}")
        if st.button(f"Remove {feed['source']}", key=f"remove_{feed['url']}"):
            remove_feed(feed['url'])

    if st.button("Refresh Feeds"):
        fetch_feed_data()


search_query = st.text_input("Search Feed Items")

# Display summaries at the top
if st.session_state.summaries:
    st.header("Ergebnisse")
    st.write("LLM: " + st.session_state.llm) 
    for i, summary in enumerate(st.session_state.summaries):
        if summary_length == "oneliner":
            st.write(summary['summary'] + f" [Quelle]({summary['url']})")
        else:
            st.write(summary['summary'])
            st.write(f"[Quelle]({summary['url']})")
        if st.button("Delete", key=f"delete_summary_{i}"):
            delete_summary(i)

# Display manually added URLs
if st.session_state.manual_urls:
    st.header("Manually Added URLs")
    for url in st.session_state.manual_urls:
        st.write(f"[{url}]({url})")
        if st.button(f"Remove URL", key=f"remove_manual_{url}"):
            st.session_state.manual_urls.remove(url)
            if url in st.session_state.marked_for_scraping:
                st.session_state.marked_for_scraping.remove(url)
            st.experimental_rerun()

st.header("Feed Items")
feed_data = filter_feed_items(search_query) if search_query else st.session_state.feed_data

for source, items in feed_data.items():
    st.markdown(f'<a name="{source.replace(" ", "-").lower()}"></a>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader(f"Feed: {source}")
    if items:
        for item in items:
            if item['link'] in st.session_state.deleted_entries:
                continue  # Skip deleted entries

            item_title = item['title']
            st.markdown(f"### [{item_title}]({item['link']})", unsafe_allow_html=True)
            st.text(f"Published: {item['published']}")
            st.markdown(item['summary'], unsafe_allow_html=True)  # Display full summary with HTML

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Delete", key=f"delete_{item['link']}"):
                    delete_entry(item['link'])
                    st.experimental_rerun()  # Immediately rerun the script to update the UI
            with col2:
                if item['link'] in st.session_state.marked_for_scraping:
                    st.markdown(f"<button disabled style='color: red;'>Marked for Scraping</button>", unsafe_allow_html=True)
                else:
                    if st.button("Mark for Scraping", key=f"mark_{item['link']}"):
                        mark_for_scraping(item['link'])
                        st.session_state[f"marked_{item['link']}"] = True
                        st.experimental_rerun()

# Initial fetch of feed data
if not st.session_state.feed_data:
    fetch_feed_data()
