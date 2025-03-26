import tkinter as tk
from tkinter import messagebox, ttk  # Импортируем ttk для Combobox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prophet import Prophet

# --- 1. Функция загрузки и подготовки данных ---
def load_and_prepare_data():
    try:
        # Загрузка данных
        df_train = pd.read_excel("C:\\Users\\2K\\Desktop\\train.xlsx", parse_dates=['dt'])
        df_train = df_train.rename(columns={'dt': 'ds', 'Цена на арматуру': 'y'})
        
        # Очистка данных
        def clean_data(df, value_column):
            if df[value_column].isnull().any():
                df[value_column] = df[value_column].interpolate(method='linear')
            
            Q1, Q3 = df[value_column].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[value_column] = df[value_column].clip(lower, upper)
            return df
        
        df_train = clean_data(df_train, 'y')
        
        # Настройка параметра cap
        cap = df_train['y'].max() * 1.1
        df_train['cap'] = cap
        
        return df_train, cap
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл 'train.xlsx' не найден.")
        return None, None
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
        return None, None

# --- 2. Обучение модели Prophet ---
def train_model(df_train):
    try:
        model = Prophet(
            growth='logistic',
            yearly_seasonality=True,
            seasonality_prior_scale=20,
            changepoint_prior_scale=0.5,
            changepoint_range=0.9
        )
        model.fit(df_train)
        return model
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обучить модель: {e}")
        return None

# --- 3. Функция для предсказания цен ---
def predict_price(model, cap, periods):
    try:
        future = model.make_future_dataframe(periods=periods, freq='M')
        future['cap'] = cap
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при прогнозировании: {e}")
        return None

# --- 4. Функция для показа прогноза ---
def show_forecast():
    selected_date = date_var.get()
    if not selected_date:
        messagebox.showerror("Ошибка", "Выберите дату.")
        return

    try:
        # Конвертация выбранной даты в формат datetime
        selected_date = pd.to_datetime(selected_date)

        # Поиск предсказания для выбранной даты
        forecast_row = forecast_df[forecast_df['ds'] == selected_date]
        if forecast_row.empty:
            messagebox.showerror("Ошибка", "Данные для выбранной даты отсутствуют.")
            return

        predicted_value = forecast_row['yhat'].values[0]
        lower_bound = forecast_row['yhat_lower'].values[0]
        upper_bound = forecast_row['yhat_upper'].values[0]

        result_label.config(
            text=f"Дата: {selected_date.strftime('%Y-%m-%d')}\n"
                 f"Предсказанная цена: {predicted_value:.2f} руб./тонна\n"
                 f"Доверительный интервал: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при поиске прогноза: {e}")

# --- 5. Функция для построения графика ---
def plot_forecast():
    global forecast_df

    if forecast_df is None:
        messagebox.showerror("Ошибка", "Прогноз не выполнен.")
        return

    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Исторические данные
    ax.plot(data['ds'], data['y'], label='Исторические данные', color='blue')

    # Прогноз
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Прогноз', color='red', linestyle='--')
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='red', alpha=0.2)

    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена на арматуру (руб./тонна)")
    ax.set_title("Прогноз цен на арматуру")
    ax.legend()

    # Отображение графика в интерфейсе
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

# --- 6. Основная функция для создания интерфейса ---
def main():
    global data, forecast_df, date_var, result_label, graph_frame

    # Загрузка данных
    data, cap = load_and_prepare_data()
    if data is None or cap is None:
        return

    # Обучение модели
    model = train_model(data)
    if model is None:
        return

    # Выполнение прогноза
    forecast_df = predict_price(model, cap, periods=12)  # Прогноз на 12 месяцев
    if forecast_df is None:
        return

    # Создание главного окна
    root = tk.Tk()
    root.title("Прогноз цен на арматуру")
    root.geometry("800x600")

    # Глобальные переменные
    date_var = tk.StringVar()

    # Заголовок
    title_label = tk.Label(root, text="Прогноз цен на арматуру", font=("Arial", 16))
    title_label.pack(pady=10)

    # Выбор даты
    date_label = tk.Label(root, text="Выберите дату:")
    date_label.pack(pady=5)

    # Заполнение Combobox датами
    dates = forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()
    date_combobox = ttk.Combobox(root, textvariable=date_var, values=dates, state="readonly", width=20)
    date_combobox.pack(pady=5)
    if dates:
        date_var.set(dates[0])  # Устанавливаем первую дату по умолчанию

    # Кнопка для показа прогноза
    forecast_button = tk.Button(root, text="Показать прогноз", command=show_forecast)
    forecast_button.pack(pady=10)

    # Результат прогноза
    result_label = tk.Label(root, text="", font=("Arial", 12), fg="green", justify="left")
    result_label.pack(pady=10)

    # График
    graph_frame = tk.Frame(root)
    graph_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    # Кнопка для построения графика
    plot_button = tk.Button(root, text="Построить график", command=plot_forecast)
    plot_button.pack(pady=10)

    # Запуск главного цикла
    root.mainloop()

# --- 7. Запуск приложения ---
if __name__ == "__main__":
    main()