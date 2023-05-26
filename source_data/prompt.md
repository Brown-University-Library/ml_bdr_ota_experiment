I have 100,000 items where the item has a key, with associated data, in the form of:

data = [
    'bdr:123' {
        'title': 'Test title',
        'file_type': 'PDF',
        'text': 'This is a test text',
        'keywords': ['spacecraft', 'aliens', 'moon'],
        'genre': 'scifi',
    }
]

I want to train a model in which I'd pass the model an item's data (minus "genre") -- and have it return a suggested genre. And I want to do this using a very basic neural network. 

Given a single file of python code -- what would the functions be?