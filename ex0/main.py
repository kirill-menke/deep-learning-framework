import matplotlib.pyplot as plt
from generator import ImageGenerator
from pattern import Checker, Circle, Spectrum

c0 = Checker(8, 1)
c0.draw()
#c0.show()
#plt.show()

c0 = Checker(18, 3)
c0.draw()
#c0.show()
#plt.show()

c0 = Circle(256, 50, (100, 100))
c0.draw()
#c0.show()
#plt.show()

c0 = Spectrum(256)
c0.draw()
#c0.show()
#plt.show()

g = ImageGenerator('./exercise_data', './Labels.json', 16, [32, 32], mirroring=True, shuffle=True,
                   rotation=True)
g.next()
#g.show()
#plt.show()
