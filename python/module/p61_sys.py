import sys
print(sys.path)

from test import p62_import
p62_import.sum2()

from test.p62_import import sum2
sum2()