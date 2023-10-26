from bs4 import BeautifulSoup
import requests
import pandas as pd

base_url = "https://www.proudmarycoffee.com.au/"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

product_links = set()


r = requests.get("https://www.proudmarycoffee.com.au/collections/coffee")
soup = BeautifulSoup(r.content, "lxml")
product_list = soup.find_all(
    "div",
    class_="boost-pfs-filter-product-item-inner",
)


for item in product_list:
    for link in item.find_all("a", href=True):
        # store the link starts without subscriptions options
        if "subscription" not in link["href"]:
            product_links.add(base_url + link["href"])
coffee_list = []

for link in product_links:
    r = requests.get(link, headers=headers)
    soup = BeautifulSoup(r.content, "lxml")

    name = soup.find("h1", class_="product-title h4").text.strip()
    description = soup.find("div", class_="product-description rte hidden").text.strip()
    try:
        tasting_notes = soup.find("div", class_="meta__block").text.strip()
    except (
        AttributeError
    ):  # Handle case when "percentage-line" or "line" class is not found
        tasting_notes = "no tasting notes"
    price = soup.find("span", class_="product-price__amount theme-money").text.strip()

    coffee = {
        "name": name,
        "tasting_notes": tasting_notes,
        "price": price,
        "description": description,
        "url": link,
    }
    coffee_list.append(coffee)
    print(f"Saving coffee:{name} to the database...")

df = pd.DataFrame(coffee_list)
# Save DataFrame to CSV
df.to_csv("pmc_coffee.csv", index=False)
