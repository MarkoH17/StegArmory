def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1000.0:
            break
        size /= 1000.0
    return f"{size:.{decimal_places}f} {unit}"