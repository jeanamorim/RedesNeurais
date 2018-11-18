from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(15, 1)

ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1), 1)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 3)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1), 4)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 5)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 6)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1), 7)
ds.addSample((0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0), 8)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 9)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0), 10)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 11)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0), 12)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 13)
ds.addSample((0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0), 14)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), 15)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), 16)
ds.addSample((0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0), 17)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 18)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1), 19)
ds.addSample((0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1), 20)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0), 21)
ds.addSample((0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0), 22)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 23)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1), 24)
ds.addSample((0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 25)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0), 26)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0), 27)
ds.addSample((1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0), 28)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 29)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1), 30)
ds.addSample((0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 31)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1), 32)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1), 33)
ds.addSample((0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0), 34)
ds.addSample((0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0), 35)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0), 36)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0), 37)
ds.addSample((0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0), 38)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 39)
ds.addSample((1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0), 40)
ds.addSample((1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 41)
ds.addSample((0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0), 42)
ds.addSample((0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0), 43)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), 44)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 45)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 46)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), 47)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 48)
ds.addSample((1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1), 49)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0), 50)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 51)
ds.addSample((1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 52)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0), 53)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1), 54)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0), 55)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 56)
ds.addSample((0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1), 57)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0), 58)
ds.addSample((0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1), 59)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), 60)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1), 61)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 62)
ds.addSample((0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0), 63)
ds.addSample((1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1), 64)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 65)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1), 66)
ds.addSample((0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1), 67)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 68)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1), 69)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 70)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1), 71)
ds.addSample((0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1), 72)
ds.addSample((0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0), 73)
ds.addSample((0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0), 74)
ds.addSample((1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1), 75)
ds.addSample((1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1), 76)
ds.addSample((0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0), 77)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0), 78)
ds.addSample((0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0), 79)
ds.addSample((0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0), 80)
ds.addSample((0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0), 81)
ds.addSample((0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 82)
ds.addSample((0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0), 83)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), 84)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0), 85)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), 86)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1), 87)
ds.addSample((0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1), 88)
ds.addSample((0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 89)
ds.addSample((0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0), 90)
ds.addSample((0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1), 91)
ds.addSample((0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0), 92)
ds.addSample((0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1), 93)
ds.addSample((1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0), 94)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0), 95)
ds.addSample((0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1), 96)
ds.addSample((1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1), 97)
ds.addSample((1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0), 98)
ds.addSample((1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1), 99)
ds.addSample((0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), 100)
ds.addSample((0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), 101)

nn = buildNetwork(15, 50, 1, bias=True)
trainer = BackpropTrainer(nn, ds)

for i in range(2000):
    print(trainer.train())
print('')
print('Digite um animal  que vou advinhar qual e: Pressione [1] para sim e [0] para nao')
print('')


def DeterminarAnimal(a):
    if a == 1:
        return 'aardvark'
    elif a == 2:
        return 'aardvark'
    elif a == 3:
        return 'antelope'
    elif a == 4:
        return 'bass'
    elif a == 5:
        return 'bear'
    elif a == 6:
        return 'boar'
    elif a == 7:
        return 'buffalo'
    elif a == 8:
        return 'calf'
    elif a == 9:
        return 'carp'
    elif a == 10:
        return 'catfish'
    elif a == 11:
        return 'cavy'
    elif a == 12:
        return 'cheetah'
    elif a == 13:
        return 'chicken'
    elif a == 14:
        return 'chub'
    elif a == 15:
        return 'clam'
    elif a == 16:
        return 'crab'
    elif a == 17:
        return 'crayfish'
    elif a == 18:
        return 'crow'
    elif a == 19:
        return 'deer'
    elif a == 20:
        return 'dogfish'
    elif a == 21:
        return 'dolphin'
    elif a == 22:
        return 'dove'
    elif a == 23:
        return 'duck'
    elif a == 24:
        return 'elephant'
    elif a == 25:
        return 'flamingo'
    elif a == 26:
        return 'flea'
    elif a == 27:
        return 'frog'
    elif a == 28:
        return 'frog'
    elif a == 29:
        return 'fruitbat'
    elif a == 30:
        return 'giraffe'
    elif a == 31:
        return 'girl'
    elif a == 32:
        return 'gnat'
    elif a == 33:
        return 'goat'
    elif a == 34:
        return 'gorilla'
    elif a == 35:
        return 'gull'
    elif a == 36:
        return 'haddock'
    elif a == 37:
        return 'hamster'
    elif a == 38:
        return 'hare'
    elif a == 39:
        return 'hawk'
    elif a == 40:
        return 'herring'
    elif a == 41:
        return 'honeybee'
    elif a == 42:
        return 'housefly'
    elif a == 43:
        return 'kiwi'
    elif a == 44:
        return 'ladybird'
    elif a == 45:
        return 'lark'
    elif a == 46:
        return 'leopard'
    elif a == 47:
        return 'lion'
    elif a == 48:
        return 'lobster'
    elif a == 49:
        return 'lynx'
    elif a == 50:
        return 'mink'
    elif a == 51:
        return 'mole'
    elif a == 52:
        return 'mongoose'
    elif a == 53:
        return 'moth'
    elif a == 54:
        return 'newt'
    elif a == 55:
        return 'octopus'
    elif a == 56:
        return 'opossum'
    elif a == 57:
        return 'oryx'
    elif a == 58:
        return 'ostrich'
    elif a == 59:
        return 'parakeet'
    elif a == 60:
        return 'penguin'
    elif a == 61:
        return 'pheasant'
    elif a == 62:
        return 'pike'
    elif a == 63:
        return 'piranha'
    elif a == 64:
        return 'pitviper'
    elif a == 65:
        return 'platypus'
    elif a == 66:
        return 'polecat'
    elif a == 67:
        return 'pony'
    elif a == 68:
        return 'porpoise'
    elif a == 69:
        return 'puma'
    elif a == 70:
        return 'pussycat'
    elif a == 71:
        return 'raccoon'
    elif a == 72:
        return 'reindeer'
    elif a == 73:
        return 'rhea'
    elif a == 74:
        return 'scorpion'
    elif a == 75:
        return 'seahorse'
    elif a == 76:
        return 'seal'
    elif a == 77:
        return 'sealion'
    elif a == 78:
        return 'seasnake'
    elif a == 79:
        return 'seawasp'
    elif a == 80:
        return 'skimmer'
    elif a == 81:
        return 'skua'
    elif a == 82:
        return 'slowworm'
    elif a == 83:
        return 'slug'
    elif a == 84:
        return 'sole'
    elif a == 85:
        return 'sparrow'
    elif a == 86:
        return 'squirrel'
    elif a == 87:
        return 'starfish'
    elif a == 88:
        return 'stingray'
    elif a == 89:
        return 'swan'
    elif a == 90:
        return 'termite'
    elif a == 91:
        return 'toad'
    elif a == 92:
        return 'tortoise'
    elif a == 93:
        return 'tuatara'
    elif a == 94:
        return 'tuna'
    elif a == 95:
        return 'vampire'
    elif a == 96:
        return 'vole'
    elif a == 97:
        return 'vulture'
    elif a == 98:
        return 'wallaby'
    elif a == 99:
        return 'wasp'
    elif a == 100:
        return 'wolf'
    elif a == 101:
        return 'worm'
    elif a == 102:
        return 'wren'
    else:
        return "nao encontrei o que pensou"


while True:
    temCabelo = float(input('ele tem cabelo?\n'))
    temPenas = float(input('ele tem temPenas?\n'))
    poemOvo = float(input('ele tem poemOvo?\n'))
    bebeLeite = float(input('ele tem bebeLeite?\n'))
    eleVoa = float(input('ele tem eleVoa?\n'))
    eAquatico = float(input('ele tem eAquatico?\n'))
    ePredador = float(input('ele tem ePredador?\n'))
    temDentes = float(input('ele tem temDentes?\n'))
    temEspinha = float(input('ele tem temEspinha?\n'))
    eleRespira = float(input('ele tem eleRespira?\n'))
    eVenenoso = float(input('ele tem eVenenoso?\n'))
    temBarbatanas = float(input('ele tem temBarbatanas?\n'))
    temCauda = float(input('ele tem temCauda?\n'))
    eDomestico = float(input('ele tem eDomestico?\n'))
    temCatsize = float(input('ele tem temCatsize?\n'))

    z = nn.activate((temCabelo, temPenas, poemOvo, bebeLeite, eleVoa, eAquatico, ePredador, temDentes, temEspinha,
                     eleRespira, eVenenoso, temBarbatanas, temCauda, eDomestico, temCatsize))
    animal = DeterminarAnimal(z)
    print('O animal e ->: ', animal)
