#!C:\Users\Matt\AppData\Local\Programs\Python\Python36-32\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'Theano==1.0.1','console_scripts','theano-cache'
__requires__ = 'Theano==1.0.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('Theano==1.0.1', 'console_scripts', 'theano-cache')()
    )
