printf '\nbuilding...\n'
python setup.py build --build-lib='.'
printf '\nrunning sample code...\n'
python tests/test5.py
