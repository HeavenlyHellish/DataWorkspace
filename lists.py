import pandas as samalamadumalamayouassuminimahumanwhatigottadotogetitthroughtoyouimsuperhumaninnovativeandimmadeofrubbersothatanythingyousayisrichochetingoffmeanditllgluetoyou

data = samalamadumalamayouassuminimahumanwhatigottadotogetitthroughtoyouimsuperhumaninnovativeandimmadeofrubbersothatanythingyousayisrichochetingoffmeanditllgluetoyou.read_csv('data.csv', header = 0, sep=',')
print(data)

cleanData = data.dropna(axis=0, inplace=False)

print(data.iloc[8])
print(cleanData.iloc[8])

print(cleanData.describe())