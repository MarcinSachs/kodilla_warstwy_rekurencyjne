# kodilla_warstwy_rekurencyjne

W poprzedniej części tego modułu dowiedzieliśmy się, jak działają konwolucyjne sieci neuronowe oraz jakie techniki regularyzacji można zastosować, aby nie dochodziło do przetrenowania. Poza pracą nad obrazami, mamy również problem związany z pracą nad sekwencjami, przykładem może być prognozowanie sprzedaży, praca nad tekstem czy inne dane, które są pewnym ciągiem. Do tego typu problemu stosowane są rekurencyjne sieci neuronowe (RNN - Recurrent Neural Network). Dla takich problemów nie możemy również stosować tych modeli z poprzedniej sekcji, ponieważ w standardowych sieciach neuronowych informacja płynie jedynie do przodu, nie powracając.

image
Pętla pozwala na przekazywanie wartości prognozy dalej. O rekurencyjnych sieciach neuronowych możesz myśleć jak o kopii tej samej sieci neuronowej dla późniejszych wyjść, które korzystają z wyjść poprzednich sieci neuronowych.

image
Typowymi sieciami neuronowymi rekurencyjnymi jest łańcuch komórek z funkcją aktywacji z tangensem hiperbolicznym. Przy wykorzystywaniu takich sieci informacje z dalekiej przeszłości stają się nieważne (zanikają z czasem), dlatego stosowana jest architektura sieci LSTM lub GRU.

LSTM
image
Warstwy LSTM zapamiętują informacje z przeszłości przez długi czas. Nie przejmuj się skomplikowaną budową pojedynczej komórki sieci. Liniami pokazano kierunek sygnału i drogę, którą pokonuje. Kluczowym dla komórki LSTM jest jego stan, czyli górna linia pozioma. Pierwsza pionowa linia w komórce to tzw. ‘forget gate’, jest to komórka z funkcją sigmoidalną, która decyduje o tym, jakich informacji pozbyć się z warstwy. W następnym kroku decydujemy jaką informację umieścić w stanie komórki. W ostatnim kroku jest podejmowana decyzja, co zwrócić na wyjściu sieci LSTM. Jest to połączenie informacji stanu komórki z przeszłości (funkcja tangensa hiperbolicznego) oraz wejścia tej komórki (funkcja sigmoidalna).

Podsumowując, komórka LSTM posiada dwa stany:

pamięć długoterminowa – górna pozioma linia
pamięć krótkoterminowa – dolna pozioma linia
Komórki LSTM dla pamięci długoterminowej podejmują decyzję, co powinno być zapomniane z przeszłości, jaką informację z pamięci krótkoterminowej wykorzystać do pamięci długoterminowej oraz, co najważniejsze, jaką informację z pamięci długoterminowej wykorzystać do prognozy w obecnej iteracji.

GRU
image
GRU to prostsza wersja komórki LSTM. Nie ma tutaj dwóch oddzielnych stanów, jak to było dla LSTM.

Implementacja
Do tego wykorzystamy zbiór danych airline-passengers.csv z liczbą pasażerów w miesiącach pomiędzy 1949 a 1960 rokiem.

Najpierw wczytujemy biblioteki, które będą wykorzystane w Notebooku.Możesz zatem otworzyć pusty Notebook w Google Colab.

# przetwarzanie danych
import numpy as np
import pandas as pd

# przekształcanie – normalizacja danych
from sklearn.preprocessing import MinMaxScaler

# wizualizacja
import matplotlib.pyplot as plt

# sieci neuronowe
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU

# ewaluacja modelu
from sklearn.metrics import mean_squared_error
Teraz czas na wczytanie zbioru danych, nad którym będziemy pracować.

dataset = pd.read_csv('airline-passengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset.set_index(['Month'], inplace=True)
dataset
image
Eksploracyjną analizę danych zacznijmy i zakończmy na wyświetleniu liczby pasażerów w czasie.

plt.figure(figsize=(16,9))
plt.plot(dataset['Passengers'])
plt.show()
image
Co wynika z przebiegu naszej zmiennej objaśnianej w czasie? Przede wszystkim jest sama od siebie zależna. Co to znaczy? Jeśli wyjaśnisz poprawnie zależności dla tego szeregu, to z łatwością określisz przyszłe wartości. Te zależności to:

Trend logarytmiczny – wraz z późniejszymi okresami czasu liczba pasażerów rośnie, jak i również widać wzrost w amplitudach. Jeśli weźmiesz różnicę pomiędzy maksymalną a minimalną wartością w roku, to ta różnica zwiększałaby się w czasie.
Sezonowość – widać wzrosty i spadki w tych samych miesiącach roku.
Podzielimy teraz nasz zbiór na zbiór treningowy oraz zbiór testowy. Jest to jednak szereg czasowy, a zatem kolejne obserwacje są zależne od poprzednich obserwacji. Nie możemy tasować zbioru danych, stworzymy zmienną train_size, która będzie integerem ile obserwacji trafi do zbioru treningowego, pozostałe trafią do zbioru testowego.

Sieć neuronowa jest modelem parametrycznym, a zatem powinniśmy wyskalować nasze zmienne. Wykorzystamy MinMaxScaler do tego. Co prawda obserwacje ze zbioru treningowego będą w zakresie 0-1, jednak te ze zbioru testowego wyjdą poza ten zakres. Jak inaczej można podejść do problemu, będzie to dalej opisane w zadaniu dla Ciebie.

Ponadto przypiszemy do zmiennej look_back wartość równą 3. Jest to istotne, ponieważ będzie to liczba zmiennych dla X. Prognozując wartość y, wartościami dla X będą 3 poprzednie wartości.

Do przygotowania danych i podziału wykorzystamy funkcję create_dataset.

train_size = int(len(dataset) * 0.70)
scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 3
def create_dataset(df, train_size, lback=look_back, scaler_function=scaler):
    test_size = len(df) - train_size
    train, test = df[0:train_size,:].copy(), df[train_size:len(df),:].copy()
    train = scaler_function.fit_transform(train)
    test = scaler_function.transform(test)
    X_train, X_test, y_train, y_test = [], [], [], []

    # Tworzenie zbioru treninowego
    for i in range(len(train)-lback-1):
        a = train[i:(i+lback), 0]
        X_train.append(a)
        y_train.append(train[i + lback, 0])

    # Tworzenie zbioru testowego
    for i in range(len(test)-lback-1):
        a = test[i:(i+lback), 0]
        X_test.append(a)
        y_test.append(test[i + lback, 0])

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    y_train, y_test = np.array(y_train), np.array(y_test)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_dataset(df=np.array(dataset), train_size=train_size, lback=look_back)
Wyświetlmy trzy pierwszej obserwacje dla X_train, a następnie dla y_train:

X_train[:3]
imageimage
Spójrz na pierwszą obserwację y, pojawia się ona jako zmienna objaśniająca dla drugiej i trzeciej obserwacji w X.

Teraz przyszło nam nauczyć modele poznane w tym module.

RNN
model_rnn = Sequential()
model_rnn.add(SimpleRNN(5, input_shape=(1, look_back)))
model_rnn.add(Dense(1))
model_rnn.compile(loss='mean_squared_error', optimizer='adam')
model_rnn.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
W celu oceny modelu stworzymy funkcję, której podawać będziemy model i otrzymamy wykres wartości prawdziwej wraz z prognozami oraz metrykę RMSE dla zbioru testowego.

def backtests(model, lback=look_back):

    # predykcja - train
    prediction_train = scaler.inverse_transform(model.predict(X_train))
    prediction_train = pd.Series(prediction_train.flatten(),
                                index=dataset.index[lback:len(prediction_train)+lback])

    # predykcja - test
    prediction_test = scaler.inverse_transform(model.predict(X_test))
    prediction_test = pd.Series(prediction_test.flatten(),
                                index=dataset.index[len(prediction_train)+(2*lback)+1:len(dataset)-1])

    # wizualizacja prognozy
    plt.figure(figsize=(16,9))
    plt.plot(dataset['Passengers'], color='blue', label='True values')
    plt.plot(prediction_train, color='green', label='Prediction - Train')
    plt.plot(prediction_test, color='red', label='Prediction - Test')
    plt.legend(loc='upper left')
    plt.show()

    # obliczenie RMSE
    rmse = mean_squared_error(dataset.loc[prediction_test.index, :], prediction_test) ** 0.5
    print(f'\nRMSE TEST: {rmse}')

backtests(model_rnn)
image
To, co istotne, prognoza jest opóźniona oraz niedoszacowana. W zadaniu dla Ciebie postarasz się to poprawić. Dla pierwszych obserwacji ze zbioru treningowego oraz testowego nie ma prognozy, a bierze się to stąd, że dla tych obserwacji nie ma wartości opóźnionych. Im wyższa look_back, tym więcej model potrzebuje obserwacji do wykonania prognozy.

LSTM
Uczymy model z kolejną warstwą poznaną przez Ciebie.

model_lstm = Sequential()
model_lstm.add(LSTM(5, input_shape=(1, look_back)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
backtests(model_lstm)
image
GRU
model_gru = Sequential()
model_gru.add(GRU(5, input_shape=(1, look_back)))
model_gru.add(Dense(1))
model_gru.compile(loss='mean_squared_error', optimizer='adam')
model_gru.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
backtests(model_gru)
image
Komórka GRU najlepiej sobie poradziła na zbiorze testowym.

Podsumowanie
W tym module poznaliśmy bardziej zaawansowane warstwy. Oczywiście jest to dla Ciebie dopiero początek, jednak wszelkie stworzone zaawansowane systemy AI oparte są o te algorytmy.

Zadanie: prognoza vs wartości prawdziwe
Pozostaw zbiór testowy jak obecnie. Zmień funkcję do backtestów oraz stworzenia zbiorów treningowego oraz testowego, tak aby nie ucinało pierwszych wartości ze zbioru testowego. Ze zbioru treningowego wyciągnij zbiór walidacyjny. Stwórz coś na wzór ‘siatki hiperparametrów’, tak jak dzieje się w GridSearch (Podpowiedź: Wystarczą pętle). Jako hiperparametry traktuj wartość dla zmiennej look_back oraz liczbę komórek w warstwie (units). Sprawdź zatem wszystkie architektury dla look_back z zakresu od 1 do 12 oraz units z zakresu 1 do 12. Wybierz model, który ma najniższą metrykę RMSE na zbiorze walidacyjnym i zwizualizuj prognozę vs wartości prawdziwe.

Zadanie dodatkowe
W tym module wykorzystano Min Max Scaler, jednak jak było widać na wykresach nie doszacowano prognozy. Teraz czas aby to zmienić. W pierwszym kroku zlogarytmuj wartości z datasetu (Podpowiedź: skorzystaj z np.log(dataset….). Dzięki temu zabiegowi amplituda nie będzie wzrastać z czasem. Następnie chcemy wystandaryzować zbiór, jednak zanim obliczymy stosunek obecnej wartości do wartości sprzed roku (Podpowiedź: dataset/dataset.shift(12)). Następnie już możesz skorzystać z StandardScaler.

Pamiętaj o tym, żeby podczas wizualizacji ‘cofnąć’ prognozę do wartości rzeczywistych i na tych wartościach wykonaj obliczenie RMSE.
