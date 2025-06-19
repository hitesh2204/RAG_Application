import langchain

version=langchain.__version__
if version >= '0.22.0':
    print('greater')
else:
    print('smaller')
 
