def capitalize_category(category: str):
    return " ".join([w.capitalize() for w in category.split("_")])