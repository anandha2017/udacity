print
print ("checking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")

print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")

print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")

print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")

print ("checking for pandas")
try:
    import pandas
except ImportError:
    print ("you should install pandas before continuing")

print ("checking for seaborn")
try:
    import seaborn
except ImportError:
    print ("you should install seaborn before continuing")

    
print ("checking for matplotlib.pyplot")
try:
    import matplotlib.pyplot
except ImportError:
    print ("you should install matplotlib.pyplot before continuing")

    
    