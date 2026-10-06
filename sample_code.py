def process_orders(orders):
    total = 0
    for order in orders:
        if order.get("active"):
            for item in order.get("items", []):
                if item.get("price", 0) > 0:
                    total += item["price"]
    return total
