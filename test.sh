printf '\nbuilding...\n'
python setup.py build --build-lib='seg_module'
printf '\nrunning sample code...\n'
python test.py
