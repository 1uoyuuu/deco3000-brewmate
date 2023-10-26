from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

base_url = "https://archerscoffee.com/"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

product_links = set()
for p in range(1, 4):
    r = requests.get(f"https://archerscoffee.com/collections/coffee?page={p}")

    soup = BeautifulSoup(r.content, "lxml")

    product_list = soup.find_all("product-grid-item", class_="product-grid-item")

    for item in product_list:
        for link in item.find_all("a", href=True):
            # store the link starts with products only
            product_links.add(base_url + link["href"])


coffee_list = []
for link in product_links:
    r = requests.get(link, headers=headers)
    soup = BeautifulSoup(r.content, "lxml")

    name = soup.find("h1", class_="product__title").text.strip()
    description = soup.find("div", class_="product-accordion").text.strip()
    # Use regular expressions to replace multiple spaces and line breaks with a single space
    cleaned_description = re.sub(r"\s+", " ", description)
    try:
        tasting_notes = soup.find("div", class_="product__subheading").text.strip()
    except (
        AttributeError
    ):  # Handle case when "percentage-line" or "line" class is not found
        tasting_notes = "no tasting notes"
    price = soup.find("span", class_="product__price").text.strip()

    coffee = {
        "name": name,
        "tasting_notes": tasting_notes,
        "price": price,
        "description": cleaned_description,
        "url": link,
    }
    coffee_list.append(coffee)
    print(f"Saving coffee: {name} to the database...")

df = pd.DataFrame(coffee_list)
# Save DataFrame to CSV
df.to_csv("archers_coffee.csv", index=False)
