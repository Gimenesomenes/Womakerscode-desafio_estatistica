import numpy as np
import pandas as pd 
import json


import seaborn as sns
import matplotlib.pyplot as plt
# Lendo o arquivo json

df  = pd.read_json('enem_2023.json')

# transformando o arquivo json em DataFrame

enem_dados = pd.DataFrame(df)

# Visualizando como o DataFrame é composto 
print(enem_dados.info())


# 1. Qual das disciplinas tem a maior amplitude de nota?
# Podemos fazer uma função para realizar o cálculo para gente da amplitude e coloca-lo juntamente na tabela .describe()

def enem_new(df):

    df1 = df.describe()
    df1.loc["amplitude"] = df1.loc['max'] - df1.loc['min']

    return df1

# Visualizando a tabela juntamente com a amplitude calculada.
print(enem_new(enem_dados))
print("A disciplina com maior amplitude de nota é Redação, como visto na tabela anterior.")

# 2. Qual é a média e a mediana para cada uma das disciplinas? (Lembre-se de remover todos os valores nulos 
# quando considerar a mediana)

# calculando a média para cada disciplina:

media_enem = enem_dados[['Linguagens', 'Ciências humanas', 'Ciências da natureza', 'Matemática', 'Redação']].mean()

print(f"Média para cada disciplina: \n{media_enem}")

# Calculando a mediana para cada disciplina, mas antes, excluindo os valores nulos:

mediana_enem = enem_dados[['Linguagens', 'Ciências humanas', 'Ciências da natureza', 'Matemática', 'Redação']].median(skipna=True)

print(f'Mediana para cada disciplina: \n{mediana_enem}')

# 3. Considerando o curso de Ciência da Computação da UFPE, onde o peso de cada uma das disciplinas ponderado:
# a. Redação - 2
# b. Matemática e suas Tecnologias - 4
# c. Linguagens, Códigos e suas Tecnologias - 2
# d. Ciências Humanas e suas Tecnologias - 1
# e. Ciências da Natureza e suas Tecnologias - 1
# Qual o desvio padrão e média das notas dos 500 estudantes mais bem colocados considerando esses pesos?

# copiando o DataFrame para media_ufpe
media_ufpe = enem_dados

# Contruindo a média ponderada de acordo com a informação dos pesos de cada disciplina
media_ufpe['Média Ponderada'] = (
    media_ufpe['Linguagens'] * 2 +
    media_ufpe['Ciências humanas'] * 1 +
    media_ufpe['Ciências da natureza'] * 1 +
    media_ufpe['Matemática'] * 4 + 
    media_ufpe['Redação'] * 2
) / 10

print(media_ufpe.head(500))

# Ordenando os estudantes por rank de notas para pegar os 500 primeiros
media_500 = media_ufpe.sort_values('Média Ponderada', ascending=False)
print(media_500.head(500))

# Calculando a média dos 500 primeiros:
media_500_primeiros = media_500.head(500)['Média Ponderada'].mean()

# Calculando o desvio padrão dos 500 primeiros:
desvpad_500_primeiros = media_500.head(500)['Média Ponderada'].std()

# Mostrando o resultado da média e desvio padrão:

print(f"Média das notas dos 500 primeiros estudantes: {media_500_primeiros:.2f}")

print(f"\n Desvio padrão dos 500 primeiros estudantes: {desvpad_500_primeiros:.2f}")

# 4. Se todos esses estudantes aplicassem para ciência da computação e
# existem apenas 40 vagas, qual seria a variância e média da nota dos
# estudantes que entraram no curso de ciência da computação?

ordena_40 = enem_dados.sort_values('Média Ponderada', ascending=False)

selecionados_40 = ordena_40.head(40)

print(selecionados_40)

media_selec = selecionados_40['Média Ponderada'].mean()

variancia_selec = selecionados_40['Média Ponderada'].var()

print(f"Média das notas dos estudantes que entraram no curso de Ciência da Computação: {media_selec:.2f}")

print(f"Variância das notas dos estudantes que entraram no curso de Ciência da Computação: {variancia_selec:.2f}")


# 5. Qual o valor do teto do terceiro quartil para as disciplinas de
# matemática e linguagens?

# Definindo os quartil:
q3_matematica = enem_dados['Matemática'].quantile(0.75)
q3_linguagens = enem_dados['Linguagens'].quantile(0.75)

# Arredondando os valores para cima (teto):

teto_q3_matematica = np.ceil(q3_matematica)
teto_q3_linguagens = np.ceil(q3_linguagens)

print(f"Teto do terceiro quartil para Matemática: {teto_q3_matematica}")
print(f"Teto do terceiro quartil para Linguagens: {teto_q3_linguagens}")

# 6. Faça o histograma de Redação e Linguagens, de 20 em 20 pontos.
# Podemos dizer que são histogramas simétricos, justifique e classifique
# se não assimétricas?

columns = ['Redação', 'Linguagens']
for column in columns:
    plt.figure()  # Crie uma nova figura para cada histograma
    sns.histplot(data=enem_dados, x=column, bins=np.arange(0, enem_dados[column].max() + 20, 20), kde=True)
    plt.title("Histograma: "+column)
    plt.xlabel(column)
    plt.ylabel("Frequência")
    plt.show()
print("Os histogramas são simétricos, pois desempenham bem a distribuição normal da curva unimodal.")

# 7. Agora coloque um range fixo de 0 até 1000, você ainda tem a mesma
# opinião quanto a simetria? [plt.hist(dado, bins=_, range=[0, 1000]).

columns = ['Redação', 'Linguagens']
for column in columns:
    plt.figure(figsize=(8,4))  # Crie uma nova figura para cada histograma
    sns.histplot(enem_dados[column], bins=range(0, 1001, 20), kde=True, color='red') 
    plt.title("Histograma: "+column)
    plt.xlabel(column)
    plt.ylabel("Frequência")
    plt.show()
print("Apesar de aumentar o range do gráfico fazendo com que a curva se alargue,\na distribuição continua sendo normal e simétrica.")
# 8. Faça um boxplot do quartil de todas as disciplinas de ciências da
# natureza e redação. É possível enxergar outliers? Utilize o método IQR.

# O IQR é o calculo da diferença entre o Q3 (terceiro quartil) e o Q1 (primeiro quartil) em uma distribuição de dados.

def find_outlier_iqr(dataset, colname):
# O q3 representa o valor abaixo do qual 75% dos dados estão localizados
# O q1 representa o valor abaixo do qual 25% dos dados estão localizados
    
    q3 = np.nanpercentile(dataset[colname], 75)
    q1 = np.nanpercentile(dataset[colname], 25)
    
    # calcula iqr
    iqr = q3 - q1

    # calcular outlier cutoff
    cut_off = iqr * 1.5
    #calcula margens inferiores, lower e superiores upper
    lower = q1 - cut_off
    upper = q3 + cut_off

    outliers = []

    for i in dataset[colname].values:
        if ((i > upper) or (i < lower)):
            outliers.append(i)
    print(f"\nDisciplina '{colname}':")
    print(f"IQR: {iqr}")
    print("O número de outliers encontrado foi de", len(outliers))
    print(outliers)
    return lower, upper, outliers


disciplinas = ['Ciências da natureza', 'Redação']

for disciplina in disciplinas:
    
    lower, upper, outliers = find_outlier_iqr(enem_dados, disciplina)
    plt.figure(figsize=(8, 4))
    sns.boxplot(enem_dados[disciplinas], color='cyan', showfliers=True)
    plt.title("Boxplot: " +  ' e ' .join(disciplinas))
    plt.ylabel("Notas")
    plt.show()


# 9. Remova todos os outliers e verifique se eles são passíveis de alterar a
# média nacional significativamente? (considere significativamente um
# valor acima de 5%).
    

def remove_outliers(dataset, colname, lower, upper):

    remove_data = dataset[(dataset[colname] >= lower) & (dataset[colname] <= upper)]

    return remove_data

# Média antes de remover os outliers:

media_antes = enem_dados[['Ciências da natureza', 'Redação']].mean()
clear_data = enem_dados.copy()


for disciplina in disciplinas:
    lower, upper, _ = find_outlier_iqr(clear_data, disciplina)
    clear_data = remove_outliers(clear_data, disciplina, lower, upper)

# Média depois de remover os outliers
    
media_depois = clear_data[['Ciências da natureza', 'Redação']].mean()

# Comparando as duas médias, antes e depois da remoção de outliers 

alteracao_percentual = ((media_depois - media_antes) / media_antes) * 100

# Verificando se o percentual de alteração é acima de 5%

percentual = any(abs(alteracao_percentual) > 5)


print(f"Média antes de remover os outliers:\n{media_antes}")
print(f"\nMédia depois de remover os outliers:\n{media_depois}")
print(f"\nAlteração Percentual:\n{alteracao_percentual}")
print(f"\nAlteração significativa é maior que 5%? \n{percentual}")
print("No entanto a remoção de outliers não é passível de alterar a média nacional\nsignificativamente.")


# 10.  Considerando valores nulos, tente encontrar qual seria a melhor medida
# de tendência que pode substituir as notas nulas. Média, moda ou
# mediana? Substitua o valor por todos os três e diga qual delas altera
# menos a média geral e o desvio padrão.


notas = enem_dados[['Linguagens', 'Ciências humanas', 'Ciências da natureza', 'Matemática', 'Redação']]

# Calculando média, moda e mediana para substituir os valores nulos

media = notas.mean()
moda = notas.mode().iloc[0]
mediana = notas.median()

print(media)
print(moda)
print(mediana)

# Substituindo os valores nulos pela media, moda e mediana

notas_media = notas.fillna(media)
notas_moda = notas.fillna(moda)
notas_mediana = notas.fillna(mediana)

# Calcular então a média geral e o desvio padrão para cada caso acima

# média
media_geral_media = notas_media.mean()
desv_pad_media = notas_media.std()

# moda

media_geral_moda = notas_moda.mean()
desv_pad_moda = notas_moda.std()

# mediana

media_geral_mediana = notas_mediana.mean()
desv_pad_mediana = notas_mediana.std()

# Comparar as alterações na média geral e no desvio padrão
alteracao_media = {
    'media': abs(media_geral_media - media),
    'moda': abs(media_geral_moda - media),
    'mediana': abs(media_geral_mediana - media)
}

alteracao_desvio_padrao = {
    'media': abs(desv_pad_media - notas.std()),
    'moda': abs(desv_pad_moda - notas.std()),
    'mediana': abs(desv_pad_mediana - notas.std())
}

print(f"\nAlteração na média geral: {alteracao_media}")
print(f"\nAlteração no desvio padrão: \n{alteracao_desvio_padrao}")
print("\n\nChegando a conclusão de que a média e a mediana são valores que alteram menos a média geral dos dados\nsendo ideais para substituir em casos de valores nulos.")




    


    

        
