def validate_totals(extracted_data):
    subtotal = sum([item["price"] for item in extracted_data["items"]])
    expected_total = subtotal + extracted_data["tax"]
    return abs(expected_total - extracted_data["total"]) < 2

if __name__ == "__main__":
    sample = {
        "items": [{"name":"Milk","price":50},{"name":"Bread","price":40}],
        "subtotal":90, "tax":9, "total":99
    }
    print("Valid bill:", validate_totals(sample))
