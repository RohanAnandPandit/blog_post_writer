from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import streamlit as st
from firecrawl import FirecrawlApp

app = FirecrawlApp(api_key=st.secrets.firecrawl.api_key)

class Website(BaseModel):
    url: str

@function_tool
def scrape_website(website: Website):
    """Scrape data from the website
    Args:
      - url: the URL of the website
    """
    print(f"scrape_website {website.url}")
    try:
        return app.scrape_url(website.url, params={'formats': ['markdown']})
    except Exception as e:
        print(e)

INSTRUCTIONS = """
Recommend trending topics for blog posts based on the company website.
"""
topic_recommendation_agent = Agent("Topic Recommendation Agent", 
                                   model='gpt-4o', 
                                   instructions=INSTRUCTIONS, 
                                   tools=[scrape_website]
                                   )
