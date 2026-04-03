def expand_query(query):
    return f"""
    Find books with similar themes, genre, and mood as:
    {query}
    
    Include: dark fantasy, complex characters, mature themes.
    """