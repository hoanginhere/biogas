from models import RegistrationCode
import random
import string

for i in range(20):
    letters = string.ascii_letters
    x =  ''.join(random.choice(letters) for i in range(30))
    code = RegistrationCode(x,)
