def format_doc(doc) -> str:
    """
    Formats a text document for embedding generation.

    Combines title and content into a single string that will be converted

    into an embedding vector. The title is included to improve the quality

    of the semantic representation.

    Args:
    doc: Dictionary with at least the keys 'title' and 'content'.

    Returns:
    Formatted string with title and content separated by a line break.
    """
    
    return f"{doc['title']}:\n{doc['content']}"





