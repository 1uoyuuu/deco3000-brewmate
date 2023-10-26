from bs4 import BeautifulSoup
import requests
import pandas as pd

base_url = "https://threecoffee.com"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

product_links = set()
for p in range(1, 4):
    r = requests.get(f"https://threecoffee.com/collections/filter?page={p}")

    soup = BeautifulSoup(r.content, "lxml")

    product_list = soup.find_all("div", class_="grid-product__content")

    for item in product_list:
        for link in item.find_all("a", href=True):
            # store the link starts with products only
            if link["href"].startswith("/products/"):
                product_links.add(base_url + link["href"])


coffee_list = []
for link in product_links:
    r = requests.get(link, headers=headers)
    soup = BeautifulSoup(r.content, "lxml")

    name = soup.find("h1", class_="product-single__title").text.strip()
    description = soup.find(
        "div", class_="product-single__description-full"
    ).text.strip()
    try:
        tasting_note_elements = soup.find("div", class_="percentage-line").find_all(
            "div", class_="line"
        )
        tasting_notes = [
            el.find("span", class_="name").text.strip() for el in tasting_note_elements
        ]
        tasting_notes_str = ", ".join(
            tasting_notes
        )  # This will join the list into a single string, separating each note by a comma and a space.
    except (
        AttributeError
    ):  # Handle case when "percentage-line" or "line" class is not found
        tasting_notes_str = "no tasting notes"
    price = soup.find("span", class_="product__price").text.strip()

    coffee = {
        "name": name,
        "tasting_notes": tasting_notes_str,
        "price": price,
        "description": description,
        "url": link,
    }
    coffee_list.append(coffee)
    print(f"Saving coffee:{name} to the database...")

df = pd.DataFrame(coffee_list)
# Save DataFrame to CSV
df.to_csv("three_coffee.csv", index=False)
