import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


def fetch_news(
    company_name,
    from_date,
    to_date
):

    url = (
    f"https://newsapi.org/v2/everything?"
    f"q={company_name}"
    f"&from={from_date}"
    f"&to={to_date}"
    f"&language=en"
    f"&sortBy=publishedAt"
    f"&pageSize=10"
    f"&apiKey={NEWSAPI_KEY}"
)

    response = requests.get(url)

    articles = response.json().get(
        "articles",
        []
    )

    texts = []

    for a in articles:

        title = a.get("title", "")

        desc = a.get("description", "")

        texts.append(f"{title}. {desc}")

    return texts